#!/usr/bin/env python3
"""
Builds the native macOS Desktop 'System 1 HUD.app' bundle.
Features a standalone native macOS WebKit window (Cocoa + WKWebView):
  - Zero browser tab dependency: opens in its own dark-themed native desktop window.
  - Single instance protection: clicking the Dock icon brings the existing window to front,
    preventing multiple panels or duplicate processes.
  - Native menubar with shortcuts (Cmd+Q, Cmd+W, Cmd+R, Cmd+C, Cmd+V).
  - High-res Retina cyberpunk neon icon integrated via Cocoa NSWorkspace & AppIcon.icns.
"""

import os
import shutil
import subprocess
import plistlib
from PIL import Image, ImageDraw, ImageFont
from Cocoa import NSWorkspace, NSImage


def generate_icon(project_dir, icns_path):
    print("[1/4] Gerando ícone de alta resolução (1024x1024)...")
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 44
    r = 215

    # Brilho externo neon ciano
    for glow in range(24, 0, -2):
        alpha = int(45 * (1 - glow / 24))
        draw.rounded_rectangle(
            [margin - glow, margin - glow, size - margin + glow, size - margin + glow],
            radius=r + glow,
            fill=(56, 189, 248, alpha),
        )

    # Base escura com borda ciano
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=r,
        fill=(8, 12, 22, 255),
        outline=(56, 189, 248, 255),
        width=18,
    )

    # Borda secundária interna
    draw.rounded_rectangle(
        [margin + 18, margin + 18, size - margin - 18, size - margin - 18],
        radius=r - 18,
        outline=(15, 23, 42, 255),
        width=8,
    )

    # Raio elétrico neon
    bolt = [
        (540, 150),
        (330, 520),
        (490, 520),
        (430, 830),
        (720, 450),
        (550, 450),
        (630, 150),
    ]

    for g in range(32, 0, -3):
        alpha = int(35 * (1 - g / 32))
        glow_bolt = [(x + (g if x > 512 else -g), y + (g if y > 490 else -g)) for x, y in bolt]
        draw.polygon(glow_bolt, fill=(56, 189, 248, alpha))

    draw.polygon(bolt, fill=(250, 204, 21, 255), outline=(56, 189, 248, 255))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 105)
    except Exception:
        font = ImageFont.load_default()

    draw.text((512, 850), "S1 HUD", fill=(56, 189, 248, 255), anchor="ms", font=font)

    iconset_dir = os.path.join(project_dir, "tmp_hud.iconset")
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
    os.makedirs(iconset_dir, exist_ok=True)

    for s in [16, 32, 128, 256, 512]:
        img.resize((s, s), Image.Resampling.LANCZOS).save(f"{iconset_dir}/icon_{s}x{s}.png")
        img.resize((s * 2, s * 2), Image.Resampling.LANCZOS).save(f"{iconset_dir}/icon_{s}x{s}@2x.png")

    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
    shutil.rmtree(iconset_dir)


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(project_dir, "System 1 HUD.app")
    icns_path = os.path.join(project_dir, "System1HUD.icns")
    main_m_path = os.path.join(project_dir, "scripts/macos_app/main.m")
    user_apps_path = os.path.expanduser("~/Applications/System 1 HUD.app")
    global_apps_path = "/Applications/System 1 HUD.app"

    # 1. Ícone
    if not os.path.exists(icns_path):
        generate_icon(project_dir, icns_path)
    else:
        print("[1/4] Ícone System1HUD.icns existente verificado.")

    # 2. Estrutura do bundle .app
    print("[2/4] Criando estrutura do Bundle e compilando App Desktop Nativo (Cocoa + WebKit)...")
    if os.path.exists(app_path):
        shutil.rmtree(app_path)

    macos_dir = os.path.join(app_path, "Contents/MacOS")
    resources_dir = os.path.join(app_path, "Contents/Resources")
    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    bin_name = "System 1 HUD"
    target_bin = os.path.join(macos_dir, bin_name)

    # Compilação nativa com clang ARM64 + Cocoa + WebKit
    compile_cmd = [
        "clang",
        "-O3",
        "-arch", "arm64",
        "-framework", "Cocoa",
        "-framework", "WebKit",
        main_m_path,
        "-o", target_bin,
    ]
    subprocess.run(compile_cmd, check=True)
    os.chmod(target_bin, 0o755)

    # Copia o ícone para o bundle
    target_icns = os.path.join(resources_dir, "AppIcon.icns")
    shutil.copy(icns_path, target_icns)

    # Info.plist nativo
    info_plist = {
        "CFBundleDevelopmentRegion": "pt-BR",
        "CFBundleDisplayName": "System 1 HUD",
        "CFBundleExecutable": bin_name,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": "com.system1.hud",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "System 1 HUD",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "2.0.0",
        "CFBundleVersion": "2",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "LSUIElement": False,
        "NSAppTransportSecurity": {
            "NSAllowsLocalNetworking": True,
            "NSAllowsArbitraryLoads": True,
        },
    }
    plist_path = os.path.join(app_path, "Contents/Info.plist")
    with open(plist_path, "wb") as f:
        plistlib.dump(info_plist, f)

    pkginfo_path = os.path.join(app_path, "Contents/PkgInfo")
    with open(pkginfo_path, "wb") as f:
        f.write(b"APPL????")

    # 3. Assinatura e Ícone Cocoa
    print("[3/4] Gravando ícone nativo via NSWorkspace e assinando bundle...")
    cocoa_img = NSImage.alloc().initWithContentsOfFile_(icns_path)
    NSWorkspace.sharedWorkspace().setIcon_forFile_options_(cocoa_img, app_path, 0)
    subprocess.run(["SetFile", "-a", "C", app_path], check=False)
    subprocess.run(["xattr", "-cr", app_path], check=True)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", app_path], check=True)

    # 4. Sincroniza com /Applications e ~/Applications
    print("[4/4] Sincronizando com /Applications e registrando no LaunchServices...")
    os.makedirs(os.path.expanduser("~/Applications"), exist_ok=True)
    target_locations = [global_apps_path, user_apps_path]

    for loc in target_locations:
        try:
            if os.path.exists(loc):
                shutil.rmtree(loc)
            shutil.copytree(app_path, loc)
            NSWorkspace.sharedWorkspace().setIcon_forFile_options_(cocoa_img, loc, 0)
            subprocess.run(["SetFile", "-a", "C", loc], check=False)
            subprocess.run(["xattr", "-cr", loc], check=True)
            subprocess.run(["codesign", "--force", "--deep", "--sign", "-", loc], check=True)
        except Exception as e:
            print(f"   [aviso] Falha ao sincronizar com {loc}: {e}")

    # Registra no LaunchServices e atualiza Dock
    lsregister_bin = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
    if os.path.exists(lsregister_bin):
        for loc in [global_apps_path, user_apps_path, app_path]:
            subprocess.run([lsregister_bin, "-f", "-R", loc], check=False)

    subprocess.run(["touch", global_apps_path, user_apps_path, app_path], check=False)
    subprocess.run(["killall", "Dock"], check=False)

    print("\n" + "=" * 65)
    print("✅ APLICATIVO DESKTOP NATIVO CONSTRUÍDO COM SUCESSO!")
    print(f"   • Localização Principal: {global_apps_path}")
    print(f"   • Localização Local:     {app_path}")
    print("=" * 65)
    print("✨ Destaques do App Desktop:")
    print("   1. Janela própria e independente (Cocoa + WKWebView nativo).")
    print("   2. Nunca mais abre abas no Chrome/Safari.")
    print("   3. Instância única: clicar no Dock traz a janela existente para frente.")
    print("   4. Ao fechar a janela, o treino continua rodando em segundo plano (Dock/Cmd+1 restaura).")


if __name__ == "__main__":
    main()
