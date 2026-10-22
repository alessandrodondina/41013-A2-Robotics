"""Simulation-only GUI interlock. Not a physical or certified emergency stop."""
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

PORT = 8765

class SimulationStopped(RuntimeError):
    pass

class SafetyControl:
    """Worker-side safety poller. Fails stopped if the GUI controller disappears."""
    def __init__(self, port=PORT):
        self.url = 'http://127.0.0.1:%d/status' % port

    def check(self):
        try:
            with urlopen(self.url, timeout=0.5) as response:
                status = json.load(response)
        except Exception as exc:
            raise SimulationStopped('Safety controller unavailable: %s' % exc)
        if hasattr(self, 'on_status'):
            self.on_status(status['state'])
        if status['state'] != 'RUNNING':
            raise SimulationStopped('Simulation stop requested; task cancelled.')

    def status(self):
        try:
            with urlopen(self.url, timeout=0.5) as response:
                state = json.load(response)['state']
                if hasattr(self, 'on_status'):
                    self.on_status(state)
                return state
        except Exception:
            return 'E-STOPPED'

    def complete(self):
        from urllib.request import Request
        try:
            urlopen(Request(self.url.replace('/status','/complete'), data=b'', method='POST'), timeout=0.5).close()
        except Exception:
            pass

    def close(self):
        pass  # The persistent launcher owns the browser GUI.

PAGE = '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>MediAssist Safety</title><style>body{font-family:system-ui;background:#111827;color:#fff;max-width:540px;margin:45px auto;padding:22px}button{padding:18px;margin:7px;border:0;border-radius:10px;font-size:19px;font-weight:bold;cursor:pointer}button:disabled{opacity:.35;cursor:not-allowed}#stop{background:#e11d48;color:white;width:96%;font-size:26px}#reset{background:#f59e0b}#start{background:#10b981}#state{padding:20px;background:#374151;border-radius:10px;font-size:23px}small{color:#cbd5e1}</style></head><body><h1>MediAssist | Safety Controls</h1><div id="state">Connecting...</div><p style="font-size:14px;color:#cbd5e1">🟢 SAFE / IDLE &nbsp; 🟠 OPERATING &nbsp; 🔴 STOP / FAULT</p><button id="stop" onclick="send('stop')">EMERGENCY STOP</button><button id="reset" onclick="send('reset')">RESET / ACKNOWLEDGE</button><button id="start" onclick="send('start')">START NEW DELIVERY</button><p><small>Simulation-only interlock, not a hardware E-stop. STOP cancels the current task. RESET never causes motion. START launches a fresh Swift simulation from its initial configuration, not from the interrupted pose.</small></p><script>async function send(x){try{await fetch('/'+x,{method:'POST'});await refresh()}catch(e){document.getElementById('state').textContent='Controller disconnected'}}async function refresh(){try{let r=await fetch('/status',{cache:'no-store'});let j=await r.json();let mode=['E-STOPPED','FAULT'].includes(j.state)?'STOP / FAULT':j.state==='RUNNING'?'OPERATING':'SAFE / IDLE';let el=document.getElementById('state');el.textContent='● '+mode+' — '+j.state;el.style.color=mode==='STOP / FAULT'?'#fb7185':mode==='OPERATING'?'#fbbf24':'#4ade80';document.getElementById('reset').disabled=j.state!=='E-STOPPED';document.getElementById('start').disabled=!['READY','RESET - WAITING FOR START','COMPLETE','FAULT'].includes(j.state);document.getElementById('stop').disabled=j.state==='E-STOPPED'}catch(e){document.getElementById('state').textContent='Controller disconnected'}}setInterval(refresh,300);refresh()</script></body></html>'''

class SafetyLauncher:
    def __init__(self, port=PORT):
        self.lock = threading.RLock()
        self.state = 'READY'
        self.process = None
        self.generation = 0
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.path == '/status':
                    with controller.lock:
                        data = json.dumps({'state': controller.state}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                elif self.path == '/':
                    data = PAGE.encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                else:
                    self.send_error(404)
                    return
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_POST(self):
                with controller.lock:
                    if self.path == '/stop':
                        controller.state = 'E-STOPPED'
                        print('[SAFETY] STOP latched; worker must abort.', flush=True)
                    elif self.path == '/reset' and controller.state == 'E-STOPPED':
                        controller.state = 'RESET - WAITING FOR START'
                        print('[SAFETY] Acknowledged. No motion permitted.', flush=True)
                    elif self.path == '/complete' and controller.state == 'RUNNING':
                        controller.state = 'COMPLETE'
                    elif self.path == '/start' and controller.state in ('READY','RESET - WAITING FOR START','COMPLETE','FAULT'):
                        if controller.process is not None and controller.process.poll() is None:
                            # Keep Swift visible throughout STOP and RESET.
                            # Only START authorises closing the old scene and launching a new one.
                            controller.state = 'RESTARTING'
                            print('[SAFETY] Restart requested. Closing old scene at START.', flush=True)
                        else:
                            controller.launch_worker()
                self.send_response(204)
                self.end_headers()

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)

    def launch_worker(self):
        self.state = 'RUNNING'
        self.generation += 1
        generation = self.generation
        self.process = subprocess.Popen([sys.executable, '-m', 'tasks.mediassist_integrated', '--worker'])
        threading.Thread(target=self.watch, args=(self.process, generation), daemon=True).start()
        print('[SAFETY] Fresh delivery launched.', flush=True)

    def watch(self, process, generation):
        code = process.wait()
        with self.lock:
            if generation != self.generation:
                return
            if self.state == 'RESTARTING':
                self.launch_worker()
            elif self.state == 'RUNNING':
                self.state = 'FAULT'
                print('[SAFETY] Worker exited (%d). State: FAULT' % code, flush=True)

    def run(self):
        print('MediAssist GUI: http://127.0.0.1:%d' % PORT, flush=True)
        print('Press START NEW DELIVERY in the browser to begin.', flush=True)
        try:
            self.server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            print('\nClosing launcher.', flush=True)
        finally:
            self.server.shutdown()
            self.server.server_close()
            with self.lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
