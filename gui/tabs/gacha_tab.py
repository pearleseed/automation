"""
Gacha Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from typing import List, Dict, Any
from PIL import Image, ImageTk

from automations.gachas import GachaAutomation
from core.agent import Agent
from core.utils import get_logger

logger = get_logger(__name__)


class GachaTab(ttk.Frame):
    """Tab for Gacha Automation with visual template selection."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent)
        self.agent = agent
        self.is_running = False
        self.selected_gachas: List[Dict[str, Any]] = []
        
        # UI variables
        self.rarity_var = tk.StringVar(value="ssr")
        self.num_pulls_var = tk.StringVar(value="10")
        self.pull_type_var = tk.StringVar(value="single")
        self.templates_path_var = tk.StringVar(value="./templates")
        self.status_var = tk.StringVar(value="Ready")
        
        self._setup_ui()
        self.load_banners()

    def _setup_ui(self):
        """Setup main UI."""
        main = ttk.Frame(self)
        main.pack(fill='both', expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        right = ttk.Frame(main)
        right.pack(side='right', fill='both', padx=(5, 0))
        right.config(width=350)

        self._setup_config(left)
        self._setup_selected(right)

    def _setup_config(self, parent):
        """Setup configuration panel."""
        # Templates Path
        path_frame = ttk.LabelFrame(parent, text="Templates Folder", padding=10)
        path_frame.pack(fill='x', pady=5)
        ttk.Entry(path_frame, textvariable=self.templates_path_var, width=40).pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(path_frame, text="Browse", command=self._browse_templates, width=10).pack(side='left')
        
        # Banner Grid
        banner_frame = ttk.LabelFrame(parent, text="Gacha Banners", padding=10)
        banner_frame.pack(fill='both', expand=True, pady=5)
        
        toolbar = ttk.Frame(banner_frame)
        toolbar.pack(fill='x', pady=(0, 5))
        ttk.Label(toolbar, text="Select banners to pull:").pack(side='left')
        ttk.Button(toolbar, text="Refresh", command=self.load_banners, width=10).pack(side='right')
        
        # Scrollable banner canvas
        canvas_frame = ttk.Frame(banner_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(canvas_frame, height=300, bg='#f5f5f5', highlightthickness=1, highlightbackground='#ccc')
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        self.banner_grid = ttk.Frame(canvas)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.banner_canvas_window = canvas.create_window((0, 0), window=self.banner_grid, anchor='nw')
        self.banner_grid.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(self.banner_canvas_window, width=e.width))
        
        # Pull Configuration
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        config_frame.pack(fill='x', pady=5)
        
        # Rarity
        r = ttk.Frame(config_frame)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text="Rarity:", width=12).pack(side='left')
        ttk.Radiobutton(r, text="SSR", variable=self.rarity_var, value="ssr").pack(side='left', padx=5)
        ttk.Radiobutton(r, text="SR", variable=self.rarity_var, value="sr").pack(side='left', padx=5)
        
        # Pulls
        p = ttk.Frame(config_frame)
        p.pack(fill='x', pady=2)
        ttk.Label(p, text="Pulls:", width=12).pack(side='left')
        ttk.Spinbox(p, from_=1, to=100, textvariable=self.num_pulls_var, width=10).pack(side='left', padx=5)
        ttk.Radiobutton(p, text="Single", variable=self.pull_type_var, value="single").pack(side='left', padx=5)
        ttk.Radiobutton(p, text="Multi (10x)", variable=self.pull_type_var, value="multi").pack(side='left')
        
        # Buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill='x', pady=10)
        self.start_btn = ttk.Button(btn_frame, text=" Start Gacha", command=self._start, style='Accent.TButton', width=20)
        self.start_btn.pack(side='left', padx=5, ipadx=15, ipady=10)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state='disabled', width=12)
        self.stop_btn.pack(side='left', padx=5, ipadx=15, ipady=10)

    def _setup_selected(self, parent):
        """Setup selected gachas panel."""
        frame = ttk.LabelFrame(parent, text="Selected Gachas", padding=10)
        frame.pack(fill='both', expand=True, pady=5)
        
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill='both', expand=True)
        
        self.listbox = tk.Listbox(list_frame, height=15, font=('', 9))
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=5)
        ttk.Button(controls, text="Remove", command=self._remove, width=10).pack(side='left', padx=2)
        ttk.Button(controls, text="Clear", command=self._clear, width=10).pack(side='left', padx=2)
        ttk.Button(controls, text="Edit", command=self._edit, width=10).pack(side='left', padx=2)
        
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill='x', pady=5)
        ttk.Label(status_frame, textvariable=self.status_var, font=('', 9), wraplength=320).pack(fill='x')

    def _browse_templates(self):
        """Browse templates folder."""
        d = filedialog.askdirectory(title="Select templates folder", initialdir=self.templates_path_var.get())
        if d:
            self.templates_path_var.set(d)
            self.load_banners()

    def load_banners(self):
        """Load banner folders from templates/banners/."""
        path = self.templates_path_var.get().strip()
        banners_path = os.path.join(path, 'banners')
        
        for w in self.banner_grid.winfo_children():
            w.destroy()
        
        # Check if folder exists
        if not os.path.exists(banners_path):
            error_frame = tk.Frame(self.banner_grid, bg='#f5f5f5')
            error_frame.pack(fill='both', expand=True, pady=40)
            tk.Label(error_frame, 
                     text=f"Folder not found", 
                     font=('', 12, 'bold'),
                     foreground='red', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text=f"{banners_path}", 
                     font=('', 10),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text="Please select a valid templates folder", 
                     font=('', 9),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            self.status_var.set("Folder not found")
            return
        
        # Find all banner folders
        try:
            banner_folders = [d for d in os.listdir(banners_path) 
                             if os.path.isdir(os.path.join(banners_path, d)) and not d.startswith('.')]
        except Exception as e:
            error_frame = tk.Frame(self.banner_grid, bg='#f5f5f5')
            error_frame.pack(fill='both', expand=True, pady=40)
            tk.Label(error_frame, 
                     text=f"Cannot access folder", 
                     font=('', 12, 'bold'),
                     foreground='red', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text=f"{banners_path}", 
                     font=('', 10),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text=f"Error: {str(e)}", 
                     font=('', 9),
                     foreground='red', bg='#f5f5f5', justify='center').pack(pady=5)
            self.status_var.set("Cannot access folder")
            return
        
        if not banner_folders:
            error_frame = tk.Frame(self.banner_grid, bg='#f5f5f5')
            error_frame.pack(fill='both', expand=True, pady=40)
            tk.Label(error_frame, 
                     text="No banners found", 
                     font=('', 12, 'bold'),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text="Create folders in templates/banners/", 
                     font=('', 10),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            tk.Label(error_frame, 
                     text="Each folder should contain image files", 
                     font=('', 9),
                     foreground='gray', bg='#f5f5f5', justify='center').pack(pady=5)
            self.status_var.set("No banners found")
            return
        
        # Configure grid columns for 2-column layout
        self.banner_grid.columnconfigure(0, weight=1, uniform='banner_col')
        self.banner_grid.columnconfigure(1, weight=1, uniform='banner_col')
        
        # Display in grid (2 columns for compact cards)
        for i, folder_name in enumerate(sorted(banner_folders)):
            folder_path = os.path.join(banners_path, folder_name)
            
            # Create card with better styling
            card = tk.Frame(self.banner_grid, relief='raised', borderwidth=2, bg='white')
            card.grid(row=i//2, column=i%2, padx=6, pady=6, sticky='nsew')
            
            # Find first image file in folder for preview
            preview_file = None
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                preview_file = os.path.join(folder_path, image_files[0])
            
            # Image container with padding
            img_frame = tk.Frame(card, bg='white')
            img_frame.pack(padx=5, pady=5, fill='x')
            
            # Display preview image (compact size)
            if preview_file:
                try:
                    img = Image.open(preview_file)
                    # Compact thumbnail size
                    img.thumbnail((140, 100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    lbl = tk.Label(img_frame, image=photo, bg='white', relief='sunken', borderwidth=1)
                    lbl.image = photo  # type: ignore
                    lbl.pack(padx=5, pady=5)
                except Exception as e:
                    error_lbl = tk.Label(img_frame, text="[Image Error]", foreground='red', bg='white', font=('', 9))
                    error_lbl.pack(padx=5, pady=5)
            else:
                no_img_lbl = tk.Label(img_frame, text="[No Image]", foreground='red', bg='white', font=('', 10, 'bold'))
                no_img_lbl.pack(padx=5, pady=5)
            
            # Display folder name (compact font, better formatting)
            name_frame = tk.Frame(card, bg='white')
            name_frame.pack(fill='x', padx=5, pady=(0, 3))
            name = folder_name if len(folder_name) <= 20 else folder_name[:17] + "..."
            name_lbl = tk.Label(name_frame, text=name, font=('', 9, 'bold'), bg='white', anchor='center')
            name_lbl.pack(fill='x')
            
            # Image count info
            info_text = f"{len(image_files)} image(s)" if image_files else "No images"
            info_lbl = tk.Label(name_frame, text=info_text, font=('', 7), bg='white', foreground='gray', anchor='center')
            info_lbl.pack(fill='x')
            
            # Check folder validity (has image files)
            has_images = len(image_files) > 0
            
            # Button frame with status indicator
            btn_frame = tk.Frame(card, bg='white')
            btn_frame.pack(pady=(0, 5), fill='x', padx=5)
            
            # Status indicator
            status_frame = tk.Frame(btn_frame, bg='white')
            status_frame.pack(side='left', padx=(0, 5))
            
            if has_images:
                status_text = "Ready"
                status_color = '#28a745'
            else:
                status_text = "No Images"
                status_color = '#ffc107'
            
            status_lbl = tk.Label(status_frame, text=status_text, font=('', 8, 'bold'), 
                                foreground=status_color, bg='white')
            status_lbl.pack()
            
            # Add button (compact, visible)
            add_btn = ttk.Button(btn_frame, text="Add", width=8,
                                command=lambda f=folder_name, p=folder_path: self._add_banner(f, p))
            add_btn.pack(side='right')
        
        self.status_var.set(f"Found {len(banner_folders)} banner folders")

    def _add_banner(self, folder_name: str, folder_path: str):
        """Add banner to selected list."""
        # Get all image files in folder
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showerror("Error", f"No image files found in folder:\n{folder_path}")
            return
        
        # If only one image file, use it directly
        if len(image_files) == 1:
            banner_file = os.path.join(folder_path, image_files[0])
        else:
            # Let user select banner file
            banner_file = filedialog.askopenfilename(
                title=f"Select banner file for {folder_name}",
                initialdir=folder_path,
                filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
            )
            if not banner_file:
                return
        
        if not os.path.exists(banner_file):
            messagebox.showerror("Error", f"Banner file not found:\n{banner_file}")
            return
        
        self.selected_gachas.append({
            'name': folder_name,
            'banner_path': banner_file,
            'banner_folder': folder_path,
            'rarity': self.rarity_var.get(),
            'num_pulls': int(self.num_pulls_var.get()),
            'pull_type': self.pull_type_var.get()
        })
        self._update_list()
        self.status_var.set(f"Added: {folder_name}")

    def _update_list(self):
        """Update selected list."""
        self.listbox.delete(0, tk.END)
        for i, g in enumerate(self.selected_gachas, 1):
            self.listbox.insert(tk.END, f"{i}. {g['name']} | {g['rarity'].upper()} | {g['pull_type']} x{g['num_pulls']}")

    def _remove(self):
        """Remove selected gacha."""
        sel = self.listbox.curselection()
        if sel:
            self.selected_gachas.pop(sel[0])
            self._update_list()

    def _clear(self):
        """Clear all."""
        if self.selected_gachas and messagebox.askyesno("Confirm", f"Remove all {len(self.selected_gachas)} gachas?"):
            self.selected_gachas.clear()
            self._update_list()

    def _edit(self):
        """Edit selected gacha."""
        sel = self.listbox.curselection()
        if not sel:
            return
        
        g = self.selected_gachas[sel[0]]
        win = tk.Toplevel(self)
        win.title(f"Edit: {g['name']}")
        win.geometry("400x200")
        win.transient(self)  # type: ignore
        
        ttk.Label(win, text=f"Editing: {g['name']}", font=('', 11, 'bold')).pack(pady=10)
        
        f = ttk.Frame(win, padding=20)
        f.pack()
        
        ttk.Label(f, text="Pulls:").grid(row=0, column=0, sticky='w', pady=5)
        pulls = tk.StringVar(value=str(g['num_pulls']))
        ttk.Spinbox(f, from_=1, to=100, textvariable=pulls, width=10).grid(row=0, column=1, pady=5)
        
        ttk.Label(f, text="Type:").grid(row=1, column=0, sticky='w', pady=5)
        ptype = tk.StringVar(value=g['pull_type'])
        pf = ttk.Frame(f)
        pf.grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(pf, text="Single", variable=ptype, value="single").pack(side='left')
        ttk.Radiobutton(pf, text="Multi", variable=ptype, value="multi").pack(side='left', padx=5)
        
        ttk.Label(f, text="Rarity:").grid(row=2, column=0, sticky='w', pady=5)
        rar = tk.StringVar(value=g['rarity'])
        rf = ttk.Frame(f)
        rf.grid(row=2, column=1, sticky='w')
        ttk.Radiobutton(rf, text="SSR", variable=rar, value="ssr").pack(side='left')
        ttk.Radiobutton(rf, text="SR", variable=rar, value="sr").pack(side='left', padx=5)
        
        def save():
            g['num_pulls'] = int(pulls.get())
            g['pull_type'] = ptype.get()
            g['rarity'] = rar.get()
            self._update_list()
            win.destroy()
        
        bf = ttk.Frame(win)
        bf.pack(pady=10)
        ttk.Button(bf, text="Save", command=save, width=12).pack(side='left', padx=5)
        ttk.Button(bf, text="Cancel", command=win.destroy, width=12).pack(side='left')

    def _start(self):
        """Start automation."""
        if not self.selected_gachas:
            messagebox.showwarning("Warning", "Add at least one banner!")
            return
        
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!")
            return
        
        total = sum(g['num_pulls'] for g in self.selected_gachas)
        msg = f"Start automation?\n\nGachas: {len(self.selected_gachas)}\nTotal Pulls: {total}"
        if not messagebox.askyesno("Confirm", msg):
            return
        
        self._set_running(True)
        
        def run():
            try:
                config = {'templates_path': self.templates_path_var.get(), 'max_scroll_attempts': 10}
                automation = GachaAutomation(self.agent, config)
                success = automation.run(self.selected_gachas)
                self.after(0, lambda: self._finished(success))
            except Exception as e:
                logger.error(f"Error: {e}")
                self.after(0, lambda: self._finished(False, str(e)))
        
        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        """Stop automation."""
        if messagebox.askyesno("Confirm", "Stop automation?"):
            self._set_running(False)

    def _set_running(self, running: bool):
        """Update UI state."""
        self.is_running = running
        self.start_btn.config(state='disabled' if running else 'normal')
        self.stop_btn.config(state='normal' if running else 'disabled')
        self.status_var.set("Running..." if running else "Ready")

    def _finished(self, success: bool, error: str = ""):
        """Handle completion."""
        self._set_running(False)
        if success:
            self.status_var.set("Completed!")
            messagebox.showinfo("Success", "Gacha automation completed!")
        else:
            self.status_var.set("Failed")
            messagebox.showerror("Error", f"Failed!\n\n{error}" if error else "Failed!")
