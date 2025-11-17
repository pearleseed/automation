"""
Gacha Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import subprocess
import platform
from typing import List, Dict, Any, Optional
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
        banner_frame = ttk.LabelFrame(parent, text="📋 Gacha Banners", padding=10)
        banner_frame.pack(fill='both', expand=True, pady=5)
        
        toolbar = ttk.Frame(banner_frame)
        toolbar.pack(fill='x', pady=(0, 5))
        ttk.Label(toolbar, text="Select banners to pull:").pack(side='left')
        ttk.Button(toolbar, text="Refresh", command=self.load_banners, width=10).pack(side='right')
        
        # Scrollable banner canvas
        canvas_frame = ttk.Frame(banner_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(canvas_frame, height=200, bg='#f5f5f5', highlightthickness=1, highlightbackground='#ccc')
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=canvas.yview)
        self.banner_grid = ttk.Frame(canvas)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.banner_canvas_window = canvas.create_window((0, 0), window=self.banner_grid, anchor='nw')
        self.banner_grid.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(self.banner_canvas_window, width=e.width))
        
        # Pull Configuration
        config_frame = ttk.LabelFrame(parent, text="⚙️ Configuration", padding=10)
        config_frame.pack(fill='x', pady=5)
        
        # Rarity
        r = ttk.Frame(config_frame)
        r.pack(fill='x', pady=2)
        ttk.Label(r, text="Rarity:", width=12).pack(side='left')
        ttk.Radiobutton(r, text="SSR ★★★★★", variable=self.rarity_var, value="ssr").pack(side='left', padx=5)
        ttk.Radiobutton(r, text="SR ★★★★", variable=self.rarity_var, value="sr").pack(side='left', padx=5)
        
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
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self._start, style='Accent.TButton', width=20)
        self.start_btn.pack(side='left', padx=5, ipady=10)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self._stop, state='disabled', width=12)
        self.stop_btn.pack(side='left', padx=5, ipady=10)

    def _setup_selected(self, parent):
        """Setup selected gachas panel."""
        frame = ttk.LabelFrame(parent, text="✅ Selected Gachas", padding=10)
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
        
        if not os.path.exists(banners_path):
            os.makedirs(banners_path, exist_ok=True)
        
        for w in self.banner_grid.winfo_children():
            w.destroy()
        
        # Find all banner folders
        banner_folders = [d for d in os.listdir(banners_path) 
                         if os.path.isdir(os.path.join(banners_path, d)) and not d.startswith('.')]
        
        if not banner_folders:
            ttk.Label(self.banner_grid, 
                     text="No banners found\nCreate folders in templates/banners/\nEach folder should contain:\n  - banner.png (required)\n  - swimsuit_*.png (required)", 
                     foreground='gray', justify='center').pack(pady=20)
            self.status_var.set("No banners found")
            return
        
        # Display in grid
        for i, folder_name in enumerate(sorted(banner_folders)):
            folder_path = os.path.join(banners_path, folder_name)
            card = ttk.Frame(self.banner_grid, relief='solid', borderwidth=1)
            card.grid(row=i//3, column=i%3, padx=5, pady=5)
            
            # Find banner.png in folder
            banner_file = None
            for ext in ['banner.png', 'banner.jpg', 'banner.jpeg']:
                test_path = os.path.join(folder_path, ext)
                if os.path.exists(test_path):
                    banner_file = test_path
                    break
            
            # Display banner image
            if banner_file:
                try:
                    img = Image.open(banner_file)
                    img.thumbnail((120, 80), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    lbl = ttk.Label(card, image=photo)
                    lbl.image = photo  # type: ignore
                    lbl.pack(padx=3, pady=3)
                except Exception:
                    ttk.Label(card, text="[Image]").pack(padx=3, pady=3)
            else:
                ttk.Label(card, text="[No Banner]", foreground='red').pack(padx=3, pady=3)
            
            # Display folder name
            name = folder_name[:17] + "..." if len(folder_name) > 20 else folder_name
            ttk.Label(card, text=name, font=('', 8)).pack()
            
            # Check folder validity
            has_banner = banner_file is not None
            swimsuit_files = [f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.lower().startswith('banner.')]
            has_swimsuit = len(swimsuit_files) > 0
            
            valid = has_banner and has_swimsuit
            hint_text = "✓" if valid else "?"
            hint_color = "green" if valid else "orange"
            
            btn_frame = ttk.Frame(card)
            btn_frame.pack(pady=3)
            ttk.Label(btn_frame, text=hint_text, foreground=hint_color, font=('', 10, 'bold')).pack(side='left', padx=(0,3))
            ttk.Button(btn_frame, text="Add", width=6, 
                      command=lambda f=folder_name, p=folder_path, b=banner_file: self._add_banner(f, p, b)).pack(side='left')
        
        self.status_var.set(f"Found {len(banner_folders)} banner folders")

    def _add_banner(self, folder_name: str, folder_path: str, banner_file: Optional[str]):
        """Add banner to selected list."""
        if not banner_file or not os.path.exists(banner_file):
            messagebox.showerror("Error", f"Banner file not found in folder:\n{folder_path}\n\nCreate a file named 'banner.png' in this folder!")
            return
        
        # Check for swimsuit files
        swimsuit_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.lower().startswith('banner.')]
        
        if not swimsuit_files:
            if messagebox.askyesno("No Swimsuit Templates", 
                f"No swimsuit templates found in:\n{folder_path}\n\nOpen folder to add swimsuit templates?"):
                if platform.system() == 'Darwin':
                    subprocess.run(['open', folder_path])
                elif platform.system() == 'Windows':
                    subprocess.run(['explorer', folder_path])
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
