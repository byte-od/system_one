#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>
#import <sys/socket.h>
#import <netinet/in.h>
#import <arpa/inet.h>
#import <unistd.h>

@interface S1AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate>
@property (strong, nonatomic) NSWindow *window;
@property (strong, nonatomic) WKWebView *webView;
@property (strong, nonatomic) NSTimer *serverCheckTimer;
@property (assign, nonatomic) NSInteger checkCount;
@property (assign, nonatomic) BOOL preserveServerOnExit;
@end

@implementation S1AppDelegate

- (BOOL)isServerRunning {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return NO;
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8050);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 80000; // 80ms timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout));
    
    int result = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    close(sock);
    return (result == 0);
}

- (void)startServerIfNeeded {
    if (![self isServerRunning]) {
        NSLog(@"[System 1 HUD] Servidor não detectado na porta 8050. Iniciando em segundo plano...");
        system("cd /Users/juliocesarreisfilho/Projects/system_one && nohup /Users/juliocesarreisfilho/Projects/system_one/.venv/bin/python -m system1_engine.hud > /tmp/system1_hud.log 2>&1 &");
    } else {
        NSLog(@"[System 1 HUD] Servidor já ativo na porta 8050.");
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // 1. Inicia backend se necessário
    [self startServerIfNeeded];
    
    // 2. Configura a janela nativa macOS
    NSRect screenRect = [[NSScreen mainScreen] visibleFrame];
    CGFloat width = 1380;
    CGFloat height = 880;
    if (width > screenRect.size.width - 40) width = screenRect.size.width - 40;
    if (height > screenRect.size.height - 40) height = screenRect.size.height - 40;
    
    NSRect frame = NSMakeRect(screenRect.origin.x + (screenRect.size.width - width) / 2,
                              screenRect.origin.y + (screenRect.size.height - height) / 2,
                              width, height);
    
    NSWindowStyleMask style = NSWindowStyleMaskTitled |
                              NSWindowStyleMaskClosable |
                              NSWindowStyleMaskMiniaturizable |
                              NSWindowStyleMaskResizable;
    
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:style
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    
    [self.window setTitle:@"System 1 Engine — Visual Control HUD"];
    [self.window setAppearance:[NSAppearance appearanceNamed:NSAppearanceNameDarkAqua]];
    [self.window setBackgroundColor:[NSColor colorWithCalibratedRed:0.03 green:0.05 blue:0.09 alpha:1.0]];
    [self.window setReleasedWhenClosed:NO];
    self.window.delegate = self;
    
    // 3. Configura WKWebView nativo com WebSockets e aceleração de hardware
    WKWebViewConfiguration *config = [[WKWebViewConfiguration alloc] init];
    config.mediaTypesRequiringUserActionForPlayback = WKAudiovisualMediaTypeNone;
    [config.preferences setValue:@YES forKey:@"developerExtrasEnabled"];
    
    self.webView = [[WKWebView alloc] initWithFrame:self.window.contentView.bounds configuration:config];
    self.webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.webView.navigationDelegate = self;
    
    // Fundo escuro transparente antes do carregamento
    [self.webView setValue:@NO forKey:@"drawsBackground"];
    [self.window.contentView addSubview:self.webView];
    
    // Splash screen com visual dark/cyberpunk durante a checagem da porta
    NSString *splashHtml = @"<!DOCTYPE html><html>"
                            "<head><meta charset='utf-8'><title>Carregando</title></head>"
                            "<body style='background:#050811;color:#38bdf8;font-family:-apple-system,BlinkMacSystemFont,monospace;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0;user-select:none;'>"
                            "<div style='font-size:32px;font-weight:900;letter-spacing:3px;margin-bottom:12px;text-shadow:0 0 20px rgba(56,189,248,0.6);'>⚡ SYSTEM 1 ENGINE</div>"
                            "<div style='color:#94a3b8;font-size:12px;letter-spacing:1px;font-family:monospace;'>INICIALIZANDO HUD CONTROL DESKTOP...</div>"
                            "</body></html>";
    [self.webView loadHTMLString:splashHtml baseURL:nil];
    
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    
    // 4. Timer com checagem rápida para carregar URL local
    self.checkCount = 0;
    self.serverCheckTimer = [NSTimer scheduledTimerWithTimeInterval:0.12
                                                             target:self
                                                           selector:@selector(checkServerAndLoad)
                                                           userInfo:nil
                                                            repeats:YES];
}

- (void)checkServerAndLoad {
    self.checkCount++;
    if ([self isServerRunning]) {
        [self.serverCheckTimer invalidate];
        self.serverCheckTimer = nil;
        NSURL *url = [NSURL URLWithString:@"http://127.0.0.1:8050"];
        NSURLRequest *request = [NSURLRequest requestWithURL:url
                                                 cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                             timeoutInterval:10.0];
        [self.webView loadRequest:request];
    } else if (self.checkCount > 60) { // ~7.2 segundos
        [self.serverCheckTimer invalidate];
        self.serverCheckTimer = nil;
        NSString *errHtml = @"<!DOCTYPE html><html><body style='background:#050811;color:#f43f5e;font-family:sans-serif;padding:40px;'>"
                            "<h2>⚠️ Falha ao conectar ao servidor HUD na porta 8050</h2>"
                            "<p style='color:#cbd5e1;'>O servidor não respondeu a tempo. Verifique o log em <code>/tmp/system1_hud.log</code></p>"
                            "<button onclick='location.reload()' style='background:#1e293b;color:#38bdf8;border:1px solid #38bdf8;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:bold;'>Tentar Novamente</button>"
                            "</body></html>";
        [self.webView loadHTMLString:errHtml baseURL:nil];
    }
}

- (void)reloadHUD:(id)sender {
    [self.webView reload];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    // Fechar a janela do HUD NÃO encerra a aplicação nem interrompe o treinamento.
    // O aplicativo permanece ativo no Dock/barra de tarefas para reconexão imediata.
    return NO;
}

- (void)showMainWindow:(id)sender {
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    return YES;
}

- (BOOL)isTrainingActive {
    if (![self isServerRunning]) return NO;
    
    NSURL *url = [NSURL URLWithString:@"http://127.0.0.1:8050/api/state"];
    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url
                                                       cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
                                                   timeoutInterval:0.35];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block BOOL isRunning = NO;
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (!error && data) {
            NSHTTPURLResponse *httpResp = (NSHTTPURLResponse *)response;
            if (httpResp.statusCode == 200) {
                id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                if ([json isKindOfClass:[NSDictionary class]]) {
                    NSDictionary *runner = json[@"runner"];
                    if ([runner isKindOfClass:[NSDictionary class]]) {
                        isRunning = [runner[@"is_running"] boolValue];
                    }
                }
            }
        }
        dispatch_semaphore_signal(sem);
    }];
    [task resume];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.45 * NSEC_PER_SEC)));
    return isRunning;
}

- (void)requestServerStop {
    NSURL *url = [NSURL URLWithString:@"http://127.0.0.1:8050/api/action"];
    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url];
    [req setHTTPMethod:@"POST"];
    [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
    [req setHTTPBody:[@"{\"action\":\"stop\"}" dataUsingEncoding:NSUTF8StringEncoding]];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        dispatch_semaphore_signal(sem);
    }];
    [task resume];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)));
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    if ([self isTrainingActive]) {
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setMessageText:@"Treinamento em Andamento"];
        [alert setInformativeText:@"Existe um treinamento ativo no momento.\n\nDeseja manter o treinamento e o servidor rodando em segundo plano, ou interromper tudo?"];
        [alert setAlertStyle:NSAlertStyleInformational];
        [alert addButtonWithTitle:@"Manter em Segundo Plano"];
        [alert addButtonWithTitle:@"Interromper Treino e Sair"];
        [alert addButtonWithTitle:@"Cancelar"];
        
        [NSApp activateIgnoringOtherApps:YES];
        NSModalResponse response = [alert runModal];
        
        if (response == NSAlertFirstButtonReturn) {
            // "Manter em Segundo Plano": encerra apenas a janela nativa sem derrubar o servidor ou treino
            self.preserveServerOnExit = YES;
            return NSTerminateNow;
        } else if (response == NSAlertSecondButtonReturn) {
            // "Interromper Treino e Sair": interrompe formalmente o treino antes de sair
            self.preserveServerOnExit = NO;
            [self requestServerStop];
            return NSTerminateNow;
        } else {
            // "Cancelar": permanece no aplicativo
            return NSTerminateCancel;
        }
    }
    return NSTerminateNow;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    if (self.preserveServerOnExit) {
        NSLog(@"[System 1 HUD] Mantendo servidor e treinamento ativos em segundo plano.");
        return;
    }
    // Encerra o servidor Python quando o app fechar sem treinamento ativo
    system("bash /Users/juliocesarreisfilho/Projects/system_one/scripts/stop_hud.sh >/dev/null 2>&1");
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        S1AppDelegate *delegate = [[S1AppDelegate alloc] init];
        app.delegate = delegate;
        
        // Menubar nativo do macOS
        NSMenu *menubar = [[NSMenu alloc] init];
        
        // App Menu
        NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:appMenuItem];
        [app setMainMenu:menubar];
        
        NSMenu *appMenu = [[NSMenu alloc] init];
        [appMenu addItemWithTitle:@"Sobre o System 1 HUD" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        [appMenu addItemWithTitle:@"Ocultar System 1 HUD" action:@selector(hide:) keyEquivalent:@"h"];
        [appMenu addItemWithTitle:@"Ocultar Outros" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
        [[appMenu itemArray].lastObject setKeyEquivalentModifierMask:(NSEventModifierFlagOption | NSEventModifierFlagCommand)];
        [appMenu addItemWithTitle:@"Mostrar Todos" action:@selector(unhideAllApplications:) keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        [appMenu addItemWithTitle:@"Encerrar System 1 HUD" action:@selector(terminate:) keyEquivalent:@"q"];
        [appMenuItem setSubmenu:appMenu];
        
        // Edit Menu (Copiar/Colar/Selecionar)
        NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:editMenuItem];
        NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Editar"];
        [editMenu addItemWithTitle:@"Desfazer" action:@selector(undo:) keyEquivalent:@"z"];
        [editMenu addItemWithTitle:@"Refazer" action:@selector(redo:) keyEquivalent:@"Z"];
        [editMenu addItem:[NSMenuItem separatorItem]];
        [editMenu addItemWithTitle:@"Recortar" action:@selector(cut:) keyEquivalent:@"x"];
        [editMenu addItemWithTitle:@"Copiar" action:@selector(copy:) keyEquivalent:@"c"];
        [editMenu addItemWithTitle:@"Colar" action:@selector(paste:) keyEquivalent:@"v"];
        [editMenu addItemWithTitle:@"Selecionar Tudo" action:@selector(selectAll:) keyEquivalent:@"a"];
        [editMenuItem setSubmenu:editMenu];
        
        // View Menu (Recarregar)
        NSMenuItem *viewMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:viewMenuItem];
        NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"Visualizar"];
        NSMenuItem *reloadItem = [[NSMenuItem alloc] initWithTitle:@"Recarregar Painel"
                                                            action:@selector(reloadHUD:)
                                                     keyEquivalent:@"r"];
        [viewMenu addItem:reloadItem];
        [viewMenuItem setSubmenu:viewMenu];
        
        // Window Menu
        NSMenuItem *windowMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:windowMenuItem];
        NSMenu *windowMenu = [[NSMenu alloc] initWithTitle:@"Janela"];
        [windowMenu addItemWithTitle:@"Exibir Painel Principal" action:@selector(showMainWindow:) keyEquivalent:@"1"];
        [windowMenu addItem:[NSMenuItem separatorItem]];
        [windowMenu addItemWithTitle:@"Fechar Janela" action:@selector(performClose:) keyEquivalent:@"w"];
        [windowMenu addItemWithTitle:@"Minimizar" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
        [windowMenu addItemWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""];
        [windowMenuItem setSubmenu:windowMenu];
        [app setWindowsMenu:windowMenu];
        
        [app run];
    }
    return 0;
}
