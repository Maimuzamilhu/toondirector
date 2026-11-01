; Windows installer. After the app folder is built, compile this with Inno Setup (free):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
#define Name "ToonDirector"
#define Version "1.0.0"

[Setup]
AppName={#Name}
AppVersion={#Version}
AppPublisher={#Name}
DefaultDirName={autopf}\{#Name}
DefaultGroupName={#Name}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=ToonDirector-Setup-{#Version}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\ToonDirector.exe
Compression=lzma2/fast
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
; lets you install on another drive, and without asking for admin rights
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Files]
Source: "..\dist\ToonDirector\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#Name}"; Filename: "{app}\ToonDirector.exe"
Name: "{autodesktop}\{#Name}"; Filename: "{app}\ToonDirector.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ToonDirector.exe"; Description: "Open {#Name}"; Flags: nowait postinstall skipifsilent
