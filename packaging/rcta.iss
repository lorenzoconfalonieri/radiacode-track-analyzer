; Inno Setup script: wraps PyInstaller's dist\rcta\ in a Windows installer.
;   iscc /DAppVersion=0.1.0 packaging\rcta.iss
; Installs per user by default (no administrator prompt); the user can choose
; all users in the wizard.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6A3B6E0C-6F55-4C1B-9E0D-5A1B6C2F7A10}
AppName=Radiacode Track Analyzer
AppVersion={#AppVersion}
AppPublisher=Lorenzo Confalonieri
AppPublisherURL=https://github.com/lorenzoconfalonieri/radiacode-track-analyzer
DefaultDirName={autopf}\Radiacode Track Analyzer
DefaultGroupName=Radiacode Track Analyzer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
SetupIconFile=..\build\icons\rcta.ico
UninstallDisplayIcon={app}\rcta.exe
OutputDir=..\dist
OutputBaseFilename=rcta-{#AppVersion}-windows-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\rcta\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Radiacode Track Analyzer"; Filename: "{app}\rcta.exe"
Name: "{autodesktop}\Radiacode Track Analyzer"; Filename: "{app}\rcta.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\rcta.exe"; Description: "{cm:LaunchProgram,Radiacode Track Analyzer}"; Flags: nowait postinstall skipifsilent
