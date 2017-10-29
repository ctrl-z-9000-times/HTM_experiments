#!/usr/bin/python3
# Written by David McDougall, 2017
"""
Graphical User Interface for labeling datasets

Outstanding Tasks
    The floodfill tool should discard probable foreground areas which arent 
        connected to any definate forground areas.
    Pending flood fill operations should remember which label they are using...
    Help message
    Show current image name
    Add a way to navigate to a specific image
"""

import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import numpy as np
import scipy
import cv2

from __init__ import Dataset

def main():
    """Creates and blocks on the GUI tool for labeling image datasets."""

    ############################################################################
    # Program and GUI Setup
    #
    # This sets up variables to hold the working state of the program.
    # This sets up the windows, frames, and menus but does not populate them.
    # Areas are populated as their functionality is defined.
    ############################################################################
    data = Dataset()

    # This is the outline user is currently tracing.
    outline = []        # List of (x,y) coordinates
    outline_ids = []    # Handles to the canvas polygon objects.

    # This is how far zoomed in/out the user currently is, size multiplier.
    scale = .25

    root = tk.Tk()
    root.wm_title("Label Tool")

    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)

    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)

    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Edit", menu=edit_menu)

    view_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="View", menu=view_menu)

    help_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Help", menu=help_menu)

    ############################################################################
    # File Menu
    ############################################################################
    def load_dataset_callback(path=None):
        if path is None:
            path = tkinter.filedialog.askdirectory()
        if not path:
            return
        data.load_dataset(path)
        redraw_canvas()
        update_label_options()
    file_menu.add_command(label="Open Dataset", command=load_dataset_callback)

    def delete_image_callback(dc=None):
        resp = tkinter.messagebox.askokcancel(
            title='Remove Image',
            message='This will move the current image to the trash can.  Procede?',
        )
        if resp:
            data.delete_current_image()
            redraw_canvas()
    file_menu.add_command(label='Delete Image', command=delete_image_callback)

    def discard_labeled_callback(dc=None):
        data.discard_labeled_data()
    file_menu.add_command(label='Discard Labeled', command=discard_labeled_callback)

    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit, accelerator='Alt-F4')

    ############################################################################
    # Edit Menu
    ############################################################################
    def add_label_dialog():
        dialog = tk.Toplevel()
        dialog.wm_title("Add Label")
        # Entry is where user types in their new label
        entry = tk.Entry(dialog)
        def entry_callback(dc=None):
            label = entry.get()
            if label and data.path:
                data.add_label_type(label)
                dialog.destroy()
                update_label_options()
        p = 4   # Padding
        entry.grid(row=0, columnspan=2, sticky='NESW', padx=p, pady=p)
        entry.focus_set()
        entry.bind("<Return>", entry_callback)
        entry.bind("<Escape>", lambda dc: dialog.destroy())

        # OK and CANCEL buttons
        ok     = tk.Button(dialog, text='Ok', command=entry_callback)
        ok.grid(row=1, column=0)
        cancel = tk.Button(dialog, text='Cancel', command=dialog.destroy)
        cancel.grid(row=1, column=1)
        root.wait_window(dialog)
    edit_menu.add_command(label='Add Label Type', command=add_label_dialog)

    def unlabel_selection(event=None):
        """Label the current selection as 'unlabeled'"""
        data.add_label_outline('unlabeled', outline)
        cancel_draw()                   # Removes the outline
        draw_labels()                   # Redraws the label image
    edit_menu.add_command(label='Unlabel', command=add_label_dialog, accelerator='u')
    root.bind("u", unlabel_selection)
    root.bind("U", unlabel_selection)

    ############################################################################
    # View Menu
    ############################################################################
    def prev_image_callback(dont_care=None):
        data.prev_image()   # Updates the database controller
        redraw_canvas()
    view_menu.add_command(label='Previous Image', command=prev_image_callback, accelerator='<-')
    root.bind("<Left>", prev_image_callback)

    def next_image_callback(dont_care=None):
        data.next_image()   # Updates the database controller
        redraw_canvas()
    view_menu.add_command(label='Next Image', command=next_image_callback, accelerator='->')
    root.bind("<Right>", next_image_callback)

    view_menu.add_separator()

    def zoom_in_callback(dontcare=None):
        nonlocal scale
        scale *= 1.25
        redraw_canvas()
    view_menu.add_command(label='Zoom in', command=zoom_in_callback, accelerator='+')
    root.bind("=", zoom_in_callback)
    root.bind("+", zoom_in_callback)

    def zoom_out_callback(dontcare=None):
        nonlocal scale
        scale /= 1.25
        redraw_canvas()
    view_menu.add_command(label='Zoom out', command=zoom_out_callback, accelerator='-')
    root.bind("-", zoom_out_callback)
    root.bind("_", zoom_out_callback)

    view_menu.add_separator()

    # These control how the existing labels are displayed to the user.
    view_label_outline = tk.BooleanVar()
    view_label_infill  = tk.BooleanVar()
    view_label_outline.set(True)
    view_label_infill.set(True)
    view_menu.add_checkbutton(label='Outline Labels', variable=view_label_outline)
    view_menu.add_checkbutton(label='Infill Labels', variable=view_label_infill)

    ############################################################################
    # Help Menu
    ############################################################################
    def show_statistics_callback(dontcare=None):
        # TODO: help button
        #           * explain how to correctly label the objects
        #           * list all keyboard shortcuts
        title = "Dataset Labeling Statistics"
        message = data.statistics()
        root.option_add('*Dialog.msg.font', 'Courier 14')
        tkinter.messagebox.showinfo(title, message)
        root.option_clear()
    help_menu.add_command(label='Statistics', command=show_statistics_callback)

    def help_menu_callback(dontcare=None):
        title = "Help"
        message = []
        message.append("TODO, Topics:")
        message.append("")
        message.append("Open Dataset")
        message.append("")
        
        message.append("File: Discard Labeled")
        discard =  "This temporarily discards images which already have labels. "
        discard += "Restart this program to see the full set of images again."
        message.append(discard)
        message.append("")

        message.append("Add Label Type")
        message.append("Outline Tool")
        message.append("Floodfill Tool")
        message.append("Cancel and Complete")
        message.append("")

        message.append("How to label:")
        instructions =  "The dataset tools use many random point samples drawn from the label image. "
        instructions += "Labels don't need to be perfectly accurate but their area should correspond "
        instructions += "to the object's area and should mostly line up.  Small specs of missing or "
        instructions += "incorrect labels are insignificant if most of the image is well labeled.  "
        message.append(instructions)
        tkinter.messagebox.showinfo(title, '\n'.join(message))
    help_menu.add_command(label='Help', command=help_menu_callback)

    ############################################################################
    # Make the image editing window
    ############################################################################
    canvas_frame = tk.Frame(root)
    canvas_frame.grid(row=1, columnspan=3, sticky='NESW')
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    canvas = tk.Canvas(canvas_frame, borderwidth=5, relief="ridge", cursor='crosshair')
    canvas.grid(row=0, column=0, sticky='NESW')
    canvas_frame.grid_columnconfigure(0, weight=1)
    canvas_frame.grid_rowconfigure(0, weight=1)

    # Horizontal and Vertical scrollbars
    xscrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    xscrollbar.grid(row=1, column=0, sticky='SEW')
    xscrollbar.config(command=canvas.xview)
    canvas.config(xscrollcommand=xscrollbar.set)
    yscrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    yscrollbar.grid(row=0, column=1, sticky='ENS')
    yscrollbar.config(command=canvas.yview)
    canvas.config(yscrollcommand=yscrollbar.set)

    # Right mouse button drags the canvas around under the mouse.
    # This way the user don't need to use the scroolbars.
    canvas.bind("<Button-3>", lambda event: canvas.scan_mark(event.x, event.y))
    canvas.bind("<B3-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1))

    def cancel_draw(event=None):
        """Discards whatever the user is currently drawing on the canvas."""
        outline.clear() # Clear the internal storage for the in-progress outline
        # Clear the outline from the screen
        for poly_id in outline_ids:
            canvas.delete(poly_id)
        outline_ids.clear()
        # Cancel the floodfill tool.
        nonlocal floodfill_mask
        floodfill_mask = None
    # cancel_button = tk.Button(toolbar, text='Cancel', command=cancel_draw)
    # cancel_button.grid(row=0, column=tb_col); tb_col += 1
    edit_menu.add_command(label='Cancel', command=cancel_draw, accelerator='Escape')
    root.bind("<Escape>", cancel_draw)

    def draw_image():
        """
        Clears the canvas and draws the current image as the background
        """
        cancel_draw()
        canvas.delete(tk.ALL)
        image = Image.open(data.current_image)
        # Rescale the image
        new_size = tuple(int(round(coord * scale)) for coord in image.size)
        image = image.resize(new_size, Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image((0, 0), image=photo, anchor='nw')
        canvas._photo_ = photo      # Save image from garbage collection
        canvas._image_ = image      # Save image from garbage collection
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def draw_labels(*args):
        draw_labels_infill()
        draw_labels_outline()
    # Redraw the labels layer everytime the user changes its settings.
    view_label_outline.trace('w', draw_labels)
    view_label_infill.trace('w', draw_labels)

    def draw_labels_infill():
        """Draws the labels over the canvas as a semi transparent layer."""
        # Delete any existing label layer before redrawing it.
        label_layer_id = getattr(canvas, '_lbl_layer_', None)
        if label_layer_id is not None:
            canvas.delete(label_layer_id)
            canvas._lbl_layer_ = None

        if view_label_infill.get():
            label = Image.open(data.current_label)
            # Rescale the label image
            new_size = tuple(int(round(coord * scale)) for coord in label.size)
            label = label.resize(new_size, Image.NEAREST)
            label = np.asarray(label)
            # Make The labels channel semi-transparent.   The label 0 is reserved
            # for unlabeled areas, make these areas fully  transparent 
            # (alpha = 0).
            alpha = np.empty(label.shape[:2], dtype=np.uint8)
            alpha.fill(96)
            alpha[np.sum(label, axis=2) == 0] = 0
            # TODO: Map the label-colors to high-visibility colors
            pass
            # Cast back to PIL
            pil_label = label[..., :3]
            pil_label = Image.frombuffer('RGBA', new_size, np.dstack([pil_label, alpha]), 'raw', 'RGBA', 0, 1)
            # Display the label layer
            photo = ImageTk.PhotoImage(pil_label)
            layer = canvas.create_image((0, 0), image=photo, anchor='nw')
            canvas._lbl_photo_ = photo          # Save image from garbage collection
            canvas._label_     = pil_label      # Save image from garbage collection
            canvas._lbl_layer_ = layer
            canvas.config(scrollregion=canvas.bbox(tk.ALL))

    def draw_labels_outline():
        """Draws the outlines of the labels over the canvas."""
        # Delete any existing label outlines first.
        label_outlines = getattr(canvas, '_lbl_outline_', [])
        for poly_id in label_outlines:
            canvas.delete(poly_id)
        canvas._lbl_outline_ = []

        if view_label_outline.get():
            cur_lbl = label_options.curselection()
            if not cur_lbl:
                # Silently fail if the user hasn't selected a label type to outline.
                return
            labels = np.array(Image.open(data.current_label), dtype=np.uint8)
            if labels.shape[2] != 4:
                # Label image missing alpha channel, outlines will not work.
                return
            selected_label = np.all(labels == label_color(), axis=2)   # Compare all color components
            selected_label = np.array(selected_label, dtype=np.uint8)  # Cast to uint8 for OpenCV
            x, contours, hierarchy = cv2.findContours(selected_label, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for edge in contours:
                edge = edge.reshape(-1, 2)
                scaled_coords = [(c0*scale, c1*scale) for c0, c1 in edge]
                edge_id = canvas.create_polygon(scaled_coords, outline='black', width=1, fill='')
                canvas._lbl_outline_.append(edge_id)

    def redraw_canvas():
        draw_image()
        draw_labels()

    def draw_outline():
        """Redraws the current outline."""
        for poly_id in outline_ids:
            canvas.delete(poly_id)
        outline_ids.clear()
        if len(outline) > 0:
            scaled_coords = [(c0*scale, c1*scale) for c0, c1 in outline]
            neon_green = "#39FF14"
            outline_ids.append(canvas.create_polygon(scaled_coords, outline=neon_green, width=3, fill=''))
            outline_ids.append(canvas.create_polygon(scaled_coords, outline='black', width=1, fill=''))

    def draw_outline_callback(event):
        """
        Callback for when user is drawing a label around an object in the image.
        User left clicks to drop a vertex of a polygon which circles the object.
            OR
        User holds down left mouse button and circles the object.
        """
        canvas_x  = canvas.canvasx(event.x) / scale
        canvas_y  = canvas.canvasy(event.y) / scale
        coords    = (canvas_x, canvas_y)
        # Don't include duplicate coordinates
        if not outline or outline[-1] != coords:
            outline.append(coords)
        draw_outline()
    canvas.bind("<Button-1>", draw_outline_callback)
    canvas.bind("<B1-Motion>", draw_outline_callback)

    # Variable holding the mask for the current flood fill operation.
    floodfill_mask = None

    def floodfill(dontcare=None):
        """
        This applies the flood fill algorithm and saves the output as 
        floodfill_mask.  This also calls draw_floodfill to display it to user
        for confirmation.
        """
        if label_color() is None:
            return

        if label_color == (0,0,0,0):
            tkinter.messagebox.showwarning(
                title='Select a label',
                message='Floodfill unlabeled is not what you wanted.',)
            return

        # Pressing the 'f' key will confirm a pending floodfill operation.
        # This way the user can spam the 'f' key and get multiple iterations
        # of floodfill.
        apply_floodfill()

        # Retrieve the data for the current image
        image       = np.array(Image.open(data.current_image))
        label_image = np.array(Image.open(data.current_label))
        # Initialize the mask to possible background.
        mask        = np.zeros(image.shape[:2], dtype=np.uint8)
        mask.fill(cv2.GC_PR_BGD)
        # Mark all areas with the current label as foreground.
        # This compares color components, then checks all components in each pixel.
        assert(label_image.shape[2] == 4)
        foreground       = np.all(label_image == label_color(), axis=2)
        mask[foreground] = cv2.GC_FGD
        # Mark all areas with a different label as background
        labeled_areas    = np.any(label_image, axis=2)
        background       = np.logical_and(np.logical_not(foreground), labeled_areas)
        mask[background] = cv2.GC_BGD
        #
        iterCount   = 1
        mode        = cv2.GC_INIT_WITH_MASK
        bgdModel    = np.zeros((1,65),np.float64)
        fgdModel    = np.zeros((1,65),np.float64)
        mask, bgdModel, fgdModel = cv2.grabCut(
                            image, mask, (0,0,1,1),
                            bgdModel  = bgdModel,
                            fgdModel  = fgdModel,
                            iterCount = iterCount,
                            mode      = mode)
        # Find the outline of the foreground segment
        foreground = np.logical_or((mask == cv2.GC_FGD), (mask == cv2.GC_PR_FGD))
        # Save the floodfill for once the user has confirmed it.
        nonlocal floodfill_mask
        floodfill_mask = np.array(foreground, dtype=np.bool)
        draw_floodfill()

    def draw_floodfill():
        assert(floodfill_mask is not None)

        # Show the suggested labeling to the user.
        ff_color   = np.array(label_color()[:3] + (0,), dtype=np.uint8)
        ff_display = np.broadcast_to(ff_color, floodfill_mask.shape[:2] + (4,))
        ff_display = np.array(ff_display, dtype=np.uint8, order='C')   # Broadcast is lazy.
        ff_display[floodfill_mask, 3] += 128

        true_size  = tuple(reversed(ff_display.shape[:2]))
        ff_pil     = Image.frombuffer('RGBA', true_size, ff_display, 'raw', 'RGBA', 0, 1)
        new_size   = tuple(int(round(dim * scale)) for dim in ff_pil.size)
        ff_pil     = ff_pil.copy().resize(new_size, Image.NEAREST)
        ff_photo   = ImageTk.PhotoImage(ff_pil)
        ff_layer   = canvas.create_image((0, 0), image=ff_photo, anchor='nw')
        canvas._ff_pil_   = ff_pil          # Save image from garbage collection
        canvas._ff_photo_ = ff_photo        # Save image from garbage collection
        # Put the layer ID where cancle_draw() will find it.
        outline_ids.append(ff_layer)

        # Draw an outline around the newly labeled areas
        mask_copy = np.array(floodfill_mask, dtype=np.uint8)
        x, contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for edge in contours:
            edge = edge.reshape(-1, 2)
            scaled_coords = [(c0*scale, c1*scale) for c0, c1 in edge]
            edge_id = canvas.create_polygon(scaled_coords, outline='black', width=1, fill='')
            outline_ids.append(edge_id)

    def apply_floodfill():
        """
        Applies the current flood fill suggestion, stored in floodfill_mask.
        Writes the new labels to file, clears the floodfill operation, and 
        redraws new labels.

        If there is no pending floodfill operation, this silently does nothing.
        """
        nonlocal floodfill_mask
        label = label_color()
        if floodfill_mask is not None and label is not None:
            data.add_label_mask(label, floodfill_mask)
            floodfill_mask = None       # Discards the (now written) flood fill data
            cancel_draw()               # Removes the outline
            draw_labels()               # Redraws the label image

    edit_menu.add_command(label='Floodfill', command=floodfill, accelerator='f')
    root.bind("f", floodfill)
    root.bind("F", floodfill)

    def finish_polygon(event=None):
        # Look up the current label
        lbl = label_color()
        if lbl is None:
            return  # No label selected, do nothing.

        apply_floodfill()

        # This polygon to the label image
        data.add_label_outline(lbl, outline)
        cancel_draw()                   # Removes the outline
        draw_labels()                   # Redraws the label image

    edit_menu.add_command(label='Complete', command=finish_polygon, accelerator='Enter/Space')
    root.bind("<space>", finish_polygon)
    root.bind("<Return>", finish_polygon)

    ##########################################
    # Show a list of available labels.
    ##########################################
    label_options = tk.Listbox(root)
    label_options.grid(row=1, column=3, sticky='NESW')

    def update_label_options():
        # Insert the labels into the list
        label_options_entries = label_options.get(0, tk.END)
        labels = data.names.values()       # dataset updates its names dict when the entry is added.
        labels = sorted(labels)            # Sort labels alphabetically
        if 'unlabeled' in labels:          # Bump special entry 'unlabeled' to top of list
            labels.remove('unlabeled')
            labels.insert(0, 'unlabeled')
        for index, label in enumerate(labels):
            if label not in label_options_entries:    # Don't make duplicate entries
                label_options.insert(index, label)
                label_options.itemconfig(index, {'bg':'white', 'fg': 'black'})

    def label_color():
        """Look up the current labels color. Returns None if no label selected."""
        label = label_options.curselection()
        if not label:       # No label selected.
            tkinter.messagebox.showwarning(
                title='No Label Selected',
                message='Please select a label from the list on the right or add a new one with the "Add Label" button.',
            )
            return None
        label_name = label_options.get(label)    # listbox widget returns index into itself...
        return next(color for color, name in data.names.items() if name == label_name)

    # load_dataset_callback('.')
    root.mainloop()


if __name__ == '__main__':
    main()
