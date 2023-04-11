import tkinter as tk
import argparse
import sys
import cv2
import json
import numpy as np
import os
sys.path.append('./pyuiutils/')
import pyuiutils.uiutils as uiutils
from tkinter import filedialog
import threading
import ttk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class SegmentationFrame(uiutils.BaseFrame):
    def __init__(self, parent, root, checkpoint_path, config_file=None):
        uiutils.BaseFrame.__init__(self, parent, root, 4, 5)
        tk.Button(self,
                  text='Load image',
                  command=self.load_image).grid(row=0,
                                                column=0,
                                                sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Undo',
                  command=self.undo).grid(row=0,
                                          column=1,
                                          sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Redo',
                  command=self.redo).grid(row=0,
                                          column=2,
                                          sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Segment',
                  command=self.process).grid(row=0,
                                             column=3,
                                             sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Save config',
                  command=self.save_config).grid(row=1,
                                                 column=0,
                                                 sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Load config',
                  command=self.load_config).grid(row=1,
                                                 column=1,
                                                 sticky=tk.W + tk.E)
        tk.Button(self,
                  text='Save image',
                  command=self.save_image).grid(row=1,
                                                column=2,
                                                sticky=tk.W + tk.E)

        # Override handle_click
        # to perform actions both on image_widget and
        # seg_widget
        self.image_widget = uiutils.ClickableImageWidget(
            self, handle_click=self.handle_click)
        self.image_widget.grid(row=2,
                               column=0,
                               columnspan=2,
                               sticky=tk.NSEW)
        self.image_widget.draw_all_points = self.draw_all_points
        
        self.seg_widget = uiutils.ImageWidget(self)
        self.seg_widget.grid(row=2,
                             column=3,
                             columnspan=2,
                             sticky=tk.NSEW)
        self.image_name = None
        self.redo_queue = []
        self.grid_rowconfigure(2, weight=1)
        self.image_receiver = None
        self.raw_image = None
        self.seg_fig = Figure()
        self.seg_canvas = FigureCanvas(self.seg_fig)
        
        if config_file is not None:
            self.load_config(config_file)


        self.forground_mode = tk.IntVar(value=1)
        self.points_mode = list()
        tk.Checkbutton(
            self, text='Foreground point', 
            variable=self.forground_mode).grid(
                row=1, column=3, sticky=tk.W + tk.E)
        # button.select()

        # Build SAM
        print("Building SAM...")
        model = build_sam(checkpoint=checkpoint_path)
        model.to(device="cuda")
        self.model = model
        self.predictor = SamPredictor(model)
        self.mask_generator = SamAutomaticMaskGenerator(model)
        print("Done initializing")

    def draw_all_points(self):
        '''Draws all the points previously selected.'''
        image_widget = self.image_widget
        image_widget.raw_image = image_widget.plain_image.copy()
        _, _, scale = image_widget.get_fitted_dimension()
        r = int(image_widget.dot_size / scale)
        color_index = 0
        for mode, (y, x) in zip(self.points_mode, image_widget.clicked_points):
            if mode == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            clicked_coords = image_widget.canvas_to_image_coordinates(y, x)
            clicked_y, clicked_x = int(clicked_coords[0]), int(
                clicked_coords[1])
            cv2.circle(image_widget.raw_image, (clicked_x, clicked_y), r,
                       color, -1)
            color_index += 1
        image_widget.redraw()


    def process(self):
        if len(self.image_widget.get_clicked_points()):
            seg_image = self.run_predictor()
        else:
            seg_image = self.run_mask_generator()
        
    def run_mask_generator(self):
        print("Running mask generator...")
        img = self.raw_image

        masks = self.mask_generator.generate(img)

        ax = self.seg_fig.gca()
        ax.imshow(img)
        for mask in masks:
            show_mask(mask['segmentation'], ax, random_color=True)
        self.seg_canvas.draw()
        im2show = np.frombuffer(self.seg_canvas.tostring_rgb(), dtype='uint8')
        width, height = self.seg_fig.get_size_inches() * self.seg_fig.get_dpi()
        width = int(width)
        height = int(height)
        im2show = im2show.reshape(height, width, 3)
        im2show = cv2.cvtColor(im2show, cv2.COLOR_RGB2BGR)

        self.seg_widget.draw_cv_image(im2show)
        print("Done")
        return im2show

    def run_predictor(self):
        print("Running predictor...")
        points = self.image_widget.get_clicked_points_in_image_coordinates()
        points = np.array([[x, y] for y, x in points], np.float32)
        # foreground points
        labels = np.array(self.points_mode)

        img = self.raw_image

        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        ax = self.seg_fig.gca()
        ax.imshow(img)
        for mask in masks:
            show_mask(mask, ax, random_color=True)
        self.seg_canvas.draw()
        im2show = np.frombuffer(self.seg_canvas.tostring_rgb(), dtype='uint8')
        width, height = self.seg_fig.get_size_inches() * self.seg_fig.get_dpi()
        width = int(width)
        height = int(height)
        im2show = im2show.reshape(height, width, 3)
        im2show = cv2.cvtColor(im2show, cv2.COLOR_RGB2BGR)

        self.seg_widget.draw_cv_image(im2show)
        print("Done")
        return im2show

    def load_image(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.image_widget.draw_new_image(img)
            self.image_name = img_name
            self.raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.seg_widget.draw_cv_image(img)

    def load_config(self, filename=None):
        if filename is None:
            filename = filedialog.askopenfilename(
                parent=self,
                filetypes=[('JSON File', '*.json')])
        if filename is not None and os.path.isfile(filename):
            with open(filename, 'r', encoding='utf-8') as infile:
                conf = json.load(infile)
                self.load_image(conf['image'])
                for c in conf['image_points']:
                    self.image_widget.push_click_image_coordinates(
                            int(c[0]), int(c[1]))
                self.set_status('Loaded from template ' + filename)

    def save_config(self):
        filename = filedialog.asksaveasfilename(
            parent=self,
            filetypes=[('JSON File', '*.json')])
        if filename is not None:
            conf = dict()
            conf['image'] = self.image_name
            conf[
                'image_points'
            ] = self.image_widget.get_clicked_points_in_image_coordinates()
            with open(filename, 'w') as outfile:
                json.dump(conf, outfile, indent=2)
                self.set_status('Saved to template ' + filename)

    def save_image(self):
        f = uiutils.ask_for_image_path_to_save(self)
        if f is not None:
            self.seg_widget.write_to_file(f)

    def undo(self):
        action = self.image_widget.pop_click()
        if action is not None:
            mode = self.points_mode.pop()
            self.redo_queue.append((action, mode))

    def redo(self):
        if len(self.redo_queue) > 0:
            action, mode = self.redo_queue.pop()
            self.points_mode.append(mode)
            self.image_widget.push_click(action[0], action[1])

    def handle_click(self, event):
        print("Handle click called")
        if not self.image_widget.has_image():
            return None
        self.points_mode.append(self.forground_mode.get())
        self.image_widget.push_click(event.y, event.x)


class SegmentAnythingUIFrame(tk.Frame):
    def __init__(self, parent, root, checkpoint_path, config_file=None):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, sticky=tk.NSEW)
        segmentation_frame = SegmentationFrame(
          notebook, root, checkpoint_path, config_file)
        notebook.add(segmentation_frame, text='Segment Anything')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run the Segment Anything GUI.')
    parser.add_argument('--model', '-m',
                        required=True,
                        help='Path to the checkpoint')
    parser.add_argument('--config', '-c',
                        help='A config file.',
                        default=None)
    args = parser.parse_args()
    root = tk.Tk()
    root.title('Segment Anything')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry('{}x{}+0+0'.format(w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app = SegmentAnythingUIFrame(root, root, args.model, args.config)
    app.grid(row=0, sticky=tk.NSEW)
    root.mainloop()
