"""The look of the app. Dusk indigo from the sets the engine draws, with the marigold of the
highlighted caption word as the single accent. Colours are named once and used everywhere."""
from pathlib import Path

CHECK = (Path(__file__).parent / "assets" / "check.svg").as_posix()

NIGHT = "#181A2C"    # window
DUSK = "#21243B"     # raised surfaces
SLATE = "#2D3150"    # inputs, hover
EDGE = "#3C4167"     # hairlines
TEXT = "#F0F1F8"
MUTED = "#A9ADC8"
MARIGOLD = "#F4B92E"  # the one accent: active things, main actions
TEAL = "#43C7B3"      # finished
CORAL = "#F36B5E"     # stopped, errors
HEAD = "Bahnschrift"
BODY = "Segoe UI Variable Text"

QSS = f"""
* {{ font-family: "{BODY}", "Segoe UI"; font-size: 14px; color: {TEXT}; }}
QMainWindow, QDialog, #page {{ background: {NIGHT}; }}
QToolTip {{ background: {SLATE}; color: {TEXT}; border: 1px solid {EDGE}; padding: 4px; }}

#rail {{ background: {DUSK}; border-right: 1px solid {EDGE}; }}
#wordmark {{ font-family: "{HEAD}"; font-size: 20px; font-weight: 600; }}
QPushButton[nav="true"] {{ text-align: left; padding: 10px 12px; border: 0; border-radius: 8px;
    background: transparent; color: {MUTED}; font-size: 15px; }}
QPushButton[nav="true"]:hover {{ background: {SLATE}; color: {TEXT}; }}
QPushButton[nav="true"]:checked {{ background: {SLATE}; color: {MARIGOLD}; font-weight: 600; }}

QLabel[role="h1"] {{ font-family: "{HEAD}"; font-size: 30px; font-weight: 600; }}
QLabel[role="h2"] {{ font-family: "{HEAD}"; font-size: 19px; font-weight: 600; }}
QLabel[role="h3"] {{ font-size: 15px; font-weight: 600; }}
QLabel[role="display"] {{ font-family: "{HEAD}"; font-size: 52px; font-weight: 600; color: {MARIGOLD}; }}
QLabel[role="stat"] {{ font-family: "{HEAD}"; font-size: 18px; font-weight: 600; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="small"] {{ color: {MUTED}; font-size: 12.5px; }}
QLabel[role="error"] {{ color: {CORAL}; }}
QLabel[role="good"] {{ color: {TEAL}; }}
QLabel[role="field"] {{ color: {MUTED}; font-size: 13px; }}

QFrame[panel="true"] {{ background: {DUSK}; border: 1px solid {EDGE}; border-radius: 12px; }}
QFrame#videoCard:hover, QFrame#videoCard:focus {{ border-color: {MARIGOLD}; }}
QFrame#railJob {{ background: {SLATE}; border-radius: 10px; }}
QFrame#railJob:hover {{ background: {EDGE}; }}
QFrame[row="true"] {{ background: transparent; border: 0; border-bottom: 1px solid {EDGE}; }}
QFrame#popover {{ background: {DUSK}; border: 1px solid {EDGE}; border-radius: 12px; }}
QFrame#toast {{ background: {SLATE}; border: 1px solid {EDGE}; border-left: 3px solid {EDGE}; border-radius: 10px; }}
QFrame#toast[kind="done"] {{ border-left-color: {TEAL}; }}
QFrame#toast[kind="error"] {{ border-left-color: {CORAL}; }}
QFrame#toast[kind="info"] {{ border-left-color: {MARIGOLD}; }}

QPushButton {{ background: {SLATE}; border: 1px solid {EDGE}; border-radius: 8px; padding: 8px 16px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QPushButton:pressed {{ background: {EDGE}; }}
QPushButton:disabled {{ color: #70748F; border-color: {SLATE}; }}
QPushButton:focus {{ border-color: {MARIGOLD}; }}
QPushButton[kind="primary"] {{ background: {MARIGOLD}; color: #221A05; border: 0; font-weight: 600; padding: 10px 22px; }}
QPushButton[kind="primary"]:hover {{ background: #FFCB4D; }}
QPushButton[kind="primary"]:disabled {{ background: {SLATE}; color: #70748F; }}
QPushButton[kind="danger"] {{ color: {CORAL}; }}
QPushButton[kind="quiet"] {{ background: transparent; border: 0; color: {MUTED}; padding: 6px 8px; }}
QPushButton[kind="quiet"]:hover {{ color: {TEXT}; }}
QPushButton[seg="true"] {{ background: transparent; border: 0; border-bottom: 2px solid transparent; border-radius: 0;
    padding: 8px 14px; color: {MUTED}; font-size: 15px; }}
QPushButton[seg="true"]:checked {{ color: {MARIGOLD}; border-bottom-color: {MARIGOLD}; }}

QLineEdit, QPlainTextEdit, QComboBox {{ background: {SLATE}; border: 1px solid {EDGE};
    border-radius: 8px; padding: 8px 10px; min-height: 20px; selection-background-color: {MARIGOLD};
    selection-color: #221A05; }}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {MARIGOLD}; }}
QPlainTextEdit[big="true"] {{ font-size: 17px; padding: 14px; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QComboBox QAbstractItemView {{ background: {DUSK}; border: 1px solid {EDGE}; selection-background-color: {SLATE};
    selection-color: {MARIGOLD}; outline: 0; padding: 4px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {EDGE}; background: {SLATE}; }}
QCheckBox::indicator:checked {{ background: {MARIGOLD}; border-color: {MARIGOLD}; image: url({CHECK}); }}
QCheckBox:focus {{ color: {MARIGOLD}; }}

QProgressBar {{ background: {SLATE}; border: 0; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {MARIGOLD}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {EDGE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {MARIGOLD}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -5px 0; border-radius: 7px; background: {TEXT}; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: 0; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {EDGE}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{ height: 0; background: none; }}
QListWidget {{ background: transparent; border: 0; outline: 0; }}
QListWidget::item {{ padding: 9px 10px; border-radius: 8px; }}
QListWidget::item:hover {{ background: {SLATE}; }}
QListWidget::item:selected {{ background: {SLATE}; color: {MARIGOLD}; }}
QPlainTextEdit[log="true"] {{ background: {DUSK}; font-family: "Cascadia Mono", Consolas; font-size: 12px; color: {MUTED}; }}
QMessageBox {{ background: {DUSK}; }}
QMenu {{ background: {DUSK}; border: 1px solid {EDGE}; padding: 4px; }}
QMenu::item {{ padding: 7px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: {SLATE}; color: {MARIGOLD}; }}
"""
