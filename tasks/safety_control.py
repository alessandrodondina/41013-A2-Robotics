"""Simulation-only GUI interlock. Not a physical or certified emergency stop."""
import json
import webbrowser
from urllib.parse import urlsplit, parse_qs
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

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MediAssist Control</title>
<style>body{font-family:system-ui;background:#111827;color:#fff;max-width:640px;margin:30px auto;padding:22px}button{padding:18px;margin:7px;border:0;border-radius:10px;font-size:19px;font-weight:bold;cursor:pointer}button:disabled{opacity:.35;cursor:not-allowed}#stop{background:#e11d48;color:white;width:96%;font-size:26px}#reset{background:#f59e0b}#start{background:#10b981}#teach{background:#3b82f6;color:#fff}#state{padding:20px;background:#374151;border-radius:10px;font-size:23px}small{color:#cbd5e1}
.tabs{display:flex;gap:6px;margin:14px 7px 4px}.tabs button{flex:1;margin:0;padding:12px;font-size:16px;background:#1f2937;color:#94a3b8;border:1px solid #374151}.tabs button.on{background:#374151;color:#fff;border-color:#60a5fa}
.hide{display:none}.panel{padding:15px;background:#1f2937;border-radius:10px;margin:10px 7px}
.row{display:flex;align-items:center;gap:8px;margin:6px 0}.row label{width:52px;color:#94a3b8}.row input{flex:1}.val{width:62px;text-align:right;font-variant-numeric:tabular-nums}
.sm{padding:10px 12px;margin:3px;font-size:15px;background:#334155;color:#e2e8f0}.sm.on{background:#10b981;color:#052e1b}
.grid{display:grid;grid-template-columns:repeat(3,72px);gap:4px}.mono{font-family:ui-monospace,Consolas,monospace;font-size:14px;white-space:pre-wrap}
</style></head><body><h1>MediAssist | Control</h1><div id="state">Connecting...</div>
<button id="stop" onclick="send('stop')">EMERGENCY STOP</button>
<div class="tabs"><button id="t_op" class="on" onclick="tab('op')">Delivery &amp; Safety</button><button id="t_jog" onclick="tab('jog')">Teach / Jog</button></div>
<div id="op">
<button id="intrusion" onclick="send('intrusion')" style="background:#f59e0b;color:#111827;width:96%">SIMULATE PERSON ENTERING</button><div style="display:flex;gap:8px"><button id="scanner_left" onclick="send('scanner_left')" style="background:#334155;color:#e2e8f0;flex:1;font-size:14px">TEST RIGHT SCANNER</button><button id="scanner_right" onclick="send('scanner_right')" style="background:#334155;color:#e2e8f0;flex:1;font-size:14px">TEST LEFT SCANNER</button></div><button id="reset" onclick="send('reset')">RESET / ACKNOWLEDGE</button><div class="panel"><label for="colour">Medication: </label><select id="colour" style="padding:10px;font-size:17px"><option value="red">Red</option><option value="green">Green</option><option value="blue">Blue</option></select><label for="bed"> Destination: </label><select id="bed" style="padding:10px;font-size:17px"><option value="1">Bed 1</option><option value="2" selected>Bed 2</option><option value="3">Bed 3</option></select></div><button id="start" onclick="send('start?colour='+encodeURIComponent(document.getElementById('colour').value)+'&bed='+encodeURIComponent(document.getElementById('bed').value))">START NEW DELIVERY</button>
</div>
<div id="jog" class="hide">
<div class="panel"><small id="jhint"></small><br><button id="teach" onclick="send('teach')" style="margin:8px 0 0">OPEN TEACH SCENE</button></div>
<div class="panel"><div class="row"><label>Robot</label><span id="robots"></span></div><div id="joints"></div>
<div class="row" id="railrow"><label>Rail</label><input type="range" id="rail" step="0.01"><span class="val" id="railv"></span></div>
<button class="sm" onclick="jogc({type:'home'})">Home selected robot</button></div>
<div class="panel"><div class="row"><label>Step</label><span id="steps"></span></div>
<div class="grid"><span></span><button class="sm" onclick="cj(1,1)">+Y</button><button class="sm" onclick="cj(2,1)">+Z</button><button class="sm" onclick="cj(0,-1)">-X</button><span></span><button class="sm" onclick="cj(0,1)">+X</button><span></span><button class="sm" onclick="cj(1,-1)">-Y</button><button class="sm" onclick="cj(2,-1)">-Z</button></div>
<small>World-frame jog, tool orientation held</small></div>
<div class="panel"><div id="jstate" class="mono"></div><div id="jfault" class="mono" style="margin-top:8px"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);let drag=-1,built=false;
function tab(t){['op','jog'].forEach(x=>{$(x).classList.toggle('hide',x!==t);$('t_'+x).classList.toggle('on',x===t)})}
async function send(x){try{await fetch('/'+x,{method:'POST'});await refresh()}catch(e){$('state').textContent='Controller disconnected'}}
async function jogc(c){try{await fetch('/jog',{method:'POST',body:JSON.stringify(c)})}catch(e){}}
function cj(a,s){jogc({type:'cart',axis:a,sign:s})}
function build(){$('joints').innerHTML=[0,1,2,3,4,5].map(j=>`<div class="row"><label>J${j+1}</label><input type="range" id="j${j}" step="1" onpointerdown="drag=${j}" onpointerup="drag=-1" oninput="$('jv${j}').textContent=this.value+'°'" onchange="jogc({type:'joint',j:${j},value:+this.value})"><span class="val" id="jv${j}"></span></div>`).join('');
$('rail').onchange=e=>jogc({type:'rail',value:+e.target.value});$('rail').oninput=e=>$('railv').textContent=(+e.target.value).toFixed(2)+' m';built=true}
function renderJog(j,state){
 const live=j&&j.robots;$('jhint').textContent=!live?'No scene open. Open a teach scene, or finish a delivery - jogging is then available in that scene.':j.allowed?'Jogging enabled.':'Jogging locked while '+state+'.';
 $('teach').disabled=!['READY','RESET - WAITING FOR START','COMPLETE','FAULT'].includes(state);if(!live)return;if(!built)build();
 $('robots').innerHTML=j.robots.map((n,i)=>`<button class="sm ${i===j.sel?'on':''}" onclick="jogc({type:'robot',idx:${i}})">${n}</button>`).join('');
 $('steps').innerHTML=j.steps.map((n,i)=>`<button class="sm ${i===j.step?'on':''}" onclick="jogc({type:'step',idx:${i}})">${n}</button>`).join('');
 for(let k=0;k<6;k++){const r=$('j'+k);r.min=j.qmin[k];r.max=j.qmax[k];if(drag!==k&&document.activeElement!==r){r.value=j.q[k];$('jv'+k).textContent=j.q[k].toFixed(0)+'°'}}
 $('railrow').style.display=j.has_rail?'':'none';if(j.has_rail){const r=$('rail');r.min=j.rail_limits[0];r.max=j.rail_limits[1];if(document.activeElement!==r){r.value=j.rail;$('railv').textContent=j.rail.toFixed(2)+' m'}}
 $('jstate').textContent=`${j.robots[j.sel]}${j.busy?'  (moving)':''}\nJoints: ${j.q.map((v,i)=>'J'+(i+1)+' '+v.toFixed(0)+'°').join('  ')}\nTool:   x ${j.tool[0].toFixed(3)}  y ${j.tool[1].toFixed(3)}  z ${j.tool[2].toFixed(3)} m${j.has_rail?'  rail '+j.rail.toFixed(2)+' m':''}`;
 $('jfault').textContent='Fault: '+j.fault;$('jfault').style.color=j.fault==='none'?'#94a3b8':'#fb7185'}
async function refresh(){try{let r=await fetch('/status',{cache:'no-store'});let j=await r.json();let mode=['E-STOPPED','FAULT','ZONE BREACH','SCANNER BREACH'].includes(j.state)?'STOP / FAULT':j.state==='RUNNING'?'OPERATING':'SAFE / IDLE';let el=$('state');el.textContent='● '+mode+' — '+j.state;el.style.color=mode==='STOP / FAULT'?'#fb7185':mode==='OPERATING'?'#fbbf24':'#4ade80';$('reset').disabled=!['E-STOPPED','ZONE BREACH','SCANNER BREACH'].includes(j.state);$('intrusion').disabled=j.state!=='RUNNING';$('scanner_left').disabled=j.state!=='RUNNING'||!!j.scanner_requested;$('scanner_right').disabled=j.state!=='RUNNING'||!!j.scanner_requested;$('start').disabled=!['READY','RESET - WAITING FOR START','COMPLETE','FAULT','TEACH'].includes(j.state);$('stop').disabled=j.state==='E-STOPPED';
 let s=await fetch('/jog_state',{cache:'no-store'});renderJog(await s.json(),j.state)}catch(e){$('state').textContent='Controller disconnected'}}
setInterval(refresh,300);refresh()</script></body></html>"""

class SafetyLauncher:
    def __init__(self, port=PORT):
        self.lock = threading.RLock()
        self.state = 'READY'
        self.process = None
        self.generation = 0
        self.intrusion_requested = False
        self.scanner_requested = None
        self.order_colour = 'red'
        self.order_bed = 2
        self.next_teach = False
        self.jog_commands = []      # page -> worker
        self.jog_state = {}         # worker -> page
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.path == '/status':
                    with controller.lock:
                        data = json.dumps({'state': controller.state, 'intrusion_requested': controller.intrusion_requested, 'scanner_requested': controller.scanner_requested}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                elif self.path == '/jog_state':
                    with controller.lock:
                        data = json.dumps(controller.jog_state).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                elif self.path == '/jog_cmds':
                    with controller.lock:
                        data = json.dumps(controller.jog_commands).encode()
                        controller.jog_commands = []
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
                body = self.rfile.read(int(self.headers.get('Content-Length', 0) or 0))
                with controller.lock:
                    path = urlsplit(self.path).path
                    if path == '/jog':
                        try:
                            controller.jog_commands.append(json.loads(body or b'{}'))
                        except ValueError:
                            pass
                    elif path == '/jog_state':
                        try:
                            controller.jog_state = json.loads(body or b'{}')
                        except ValueError:
                            pass
                    elif path == '/teach' and controller.state in ('READY','RESET - WAITING FOR START','COMPLETE','FAULT'):
                        controller.next_teach = True
                        if controller.process is not None and controller.process.poll() is None:
                            controller.state = 'RESTARTING'
                            print('[TEACH] Closing old scene for a fresh teach scene.', flush=True)
                        else:
                            controller.launch_worker()
                    elif path == '/stop':
                        controller.state = 'E-STOPPED'
                        print('[SAFETY] STOP latched; worker must abort.', flush=True)
                    elif path == '/intrusion' and controller.state == 'RUNNING':
                        controller.intrusion_requested = True
                        print('[SIMULATION] Person entering light curtain.', flush=True)
                    elif path in ('/scanner_left', '/scanner_right') and controller.state == 'RUNNING':
                        controller.scanner_requested = path.split('_')[-1]
                        print('[SIMULATION] Worker approaching %s rail scanner.' % controller.scanner_requested, flush=True)
                    elif path == '/scanner_breach' and controller.state == 'RUNNING' and controller.scanner_requested:
                        controller.state = 'SCANNER BREACH'
                        print('[RAIL SCANNER] Protective field breached: stop latched.', flush=True)
                    elif path == '/reset' and controller.state in ('E-STOPPED','ZONE BREACH','SCANNER BREACH','SCANNER BREACH'):
                        controller.state = 'RESET - WAITING FOR START'
                        print('[SAFETY] Acknowledged. No motion permitted.', flush=True)
                    elif path == '/curtain_breach' and controller.state == 'RUNNING' and controller.intrusion_requested:
                        controller.state = 'ZONE BREACH'
                        print('[LIGHT CURTAIN] Beam interrupted: safety stop latched.', flush=True)
                    elif path == '/complete' and controller.state == 'RUNNING':
                        controller.state = 'COMPLETE'
                    elif path == '/start' and controller.state in ('READY','RESET - WAITING FOR START','COMPLETE','FAULT','TEACH'):
                        params = parse_qs(urlsplit(self.path).query)
                        colour = params.get('colour', ['red'])[0]
                        bed = params.get('bed', ['2'])[0]
                        if colour not in ('red','green','blue') or bed not in ('1','2','3'):
                            self.send_error(400, 'Invalid medication or bed')
                            return
                        controller.order_colour = colour
                        controller.order_bed = int(bed)
                        controller.next_teach = False
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
        self.state = 'TEACH' if self.next_teach else 'RUNNING'
        self.intrusion_requested = False
        self.scanner_requested = None
        self.jog_commands = []
        self.jog_state = {}
        self.generation += 1
        generation = self.generation
        args = [sys.executable, '-m', 'tasks.mediassist_integrated', '--worker', '--colour', self.order_colour, '--bed', str(self.order_bed)]
        if self.next_teach:
            args.append('--teach')
        self.process = subprocess.Popen(args)
        threading.Thread(target=self.watch, args=(self.process, generation), daemon=True).start()
        print('[SAFETY] Fresh delivery launched.', flush=True)

    def watch(self, process, generation):
        code = process.wait()
        with self.lock:
            if generation != self.generation:
                return
            if self.state == 'RESTARTING':
                self.launch_worker()
            elif self.state in ('RUNNING', 'TEACH'):
                self.state = 'FAULT'
                print('[SAFETY] Worker exited (%d). State: FAULT' % code, flush=True)

    def run(self):
        print('MediAssist GUI: http://127.0.0.1:%d' % PORT, flush=True)
        print('Press START NEW DELIVERY in the browser to begin.', flush=True)
        webbrowser.open('http://127.0.0.1:%d/' % PORT)
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
