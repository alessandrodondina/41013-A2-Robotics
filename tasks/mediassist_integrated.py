"""MediAssist combined demo: Kawasaki RED pickup -> transfer -> Doosan bedside delivery.
Run from project root: python3 -m tasks.mediassist_integrated
Standalone demo only; original Doosan V7 and Kawasaki scripts remain unchanged.
Collision avoidance is approximate and not certified.
"""
"""MediAssist Doosan delivery with collision-aware route and smooth upright placement and tabletop contact.
Run: python3 -m robot_models.doosan_delivery_safe_v7
This is an approximate link-centreline guard, not certified collision avoidance.
"""
import time
import random
from pathlib import Path
import numpy as np
import swift
from spatialmath import SE3
from spatialmath.base import rpy2r
from spatialgeometry import Mesh, Cuboid, Cylinder
from environment.ward import build_ward
import layout
from tasks.kawasaki_pick_place import KawasakiCell, find_object_at
from environment.sensors import RGBDCamera, PhotoelectricSensor
from perception.rgbd_detection import PinholeCamera, SimulatedRGBD, detect_containers, save_debug_image
from robot_models.doosan_m0609 import create_doosan, JOINT_XYZ, JOINT_RPY, swift_mesh_path
from tasks.safety_control import SafetyControl, SimulationStopped
from tasks.safety_equipment import SafetyEquipment

RAIL_Y, BASE_Z, CARRIAGE_Z = .75, .96, .90
TRANSFER_X, TRANSFER_Y, TRANSFER_Z = 1.65, .12, .83
TABLES = {1:(.86,1.47,.72), 2:(2.26,1.47,.72), 3:(3.66,1.47,.72)}
BOTTLE_H = .11
MESH_DIR = Path(__file__).resolve().parent.parent/'robot_models'/'meshes'/'m0609_white'
MESH_FILES = [
 ['MF0609_0_0.dae'], ['MF0609_1_0.dae'],
 ['MF0609_2_0.dae','MF0609_2_1.dae','MF0609_2_2.dae'],
 ['MF0609_3_0.dae'], ['MF0609_4_0.dae','MF0609_4_1.dae'],
 ['MF0609_5_0.dae'], ['MF0609_6_0.dae']]

def base_pose(x): return SE3(x,RAIL_Y,BASE_Z)

def arm_points(x, joints):
    """Joint-frame centres from the same URDF transforms as the visible meshes."""
    T=base_pose(x)
    points=[np.asarray(T.t)]
    for i in range(6):
        T=T*SE3.Rt(rpy2r(JOINT_RPY[i],order='zyx'),JOINT_XYZ[i])*SE3.Rz(float(joints[i]))
        points.append(np.asarray(T.t))
    return points

# Light-curtain posts on the transfer side (x, y), from 0.78 m to 1.68 m
# high. Half-width = 27.5 mm post + ~90 mm for the arm/bottle thickness.
CURTAIN_POSTS = ((0.94, 0.39), (2.36, 0.39))
POST_HALF = 0.12
POST_Z = (0.70, 1.72)

def rail_clearance(x, joints):
    """Conservative approximate link-segment vs rail-body test.
    Rail body occupies x=-.05..3.45, y=.61..89, z=.65..83.
    Added 35mm clearance. Ignore anchored base-to-shoulder segment.
    Also keeps the arm, tool and a held bottle clear of the light-curtain
    posts beside the transfer plate.
    """
    points=arm_points(x,joints)
    low=np.array([-.085,.575,.615])
    high=np.array([3.485,.925,.865])
    for link in range(2,7):
        a,b=points[link-1],points[link]
        for t in np.linspace(0,1,13):
            p=a+(b-a)*t
            if np.all(p>=low) and np.all(p<=high):
                return False, f'arm link {link} enters rail clearance zone at {np.round(p,3)}'
    segments=[(points[i-1],points[i]) for i in range(2,7)]
    segments.append((points[6],points[6]+np.array([0,0,-.17])))   # tool + bottle
    for a,b in segments:
        for t in np.linspace(0,1,13):
            p=a+(b-a)*t
            if POST_Z[0]<=p[2]<=POST_Z[1]:
                for px,py in CURTAIN_POSTS:
                    if abs(p[0]-px)<=POST_HALF and abs(p[1]-py)<=POST_HALF:
                        return False, f'arm would hit light-curtain post at ({px}, {py})'
    return True,''

def solve_ik(robot,x,target,seed):
    """Find a rail-clear endpoint from several numerical IK restarts."""
    robot.base=base_pose(x)
    goal=np.asarray(target.t,dtype=float)
    rng=np.random.default_rng(21)
    guesses=[np.asarray(seed,dtype=float),np.zeros(6)]
    guesses.extend(rng.uniform(-2.5,2.5,size=(32,6)))
    best=None
    for start in guesses:
        candidate=start.copy()
        for _ in range(350):
            error=goal-robot.fkine(candidate).t
            if np.linalg.norm(error)<.006:
                ok,_=rail_clearance(x,candidate)
                if ok:
                    score=np.linalg.norm(candidate-seed)
                    if best is None or score<best[0]: best=(score,candidate.copy())
                break
            J=robot.jacob0(candidate)[:3,:]
            dq=J.T@np.linalg.solve(J@J.T+.04**2*np.eye(3),error)
            candidate+=np.clip(dq,-.10,.10)
            candidate=np.clip(candidate,-3.0,3.0)
        if best is not None and best[0]<1.5: break
    if best is None:
        raise RuntimeError(f'No rail-clear IK target at {np.round(goal,3)}; leaving working script untouched.')
    return best[1]


def collision_free_edge(a,b):
    """Sample complete 7D rail/joint movement at small increments."""
    n=max(2,int(np.max(np.abs((b-a)/np.array([.035,.09,.09,.09,.09,.09,.09]))))+1)
    for t in np.linspace(0,1,n):
        state=a+(b-a)*t
        if not rail_clearance(state[0],state[1:])[0]: return False
    return True


def plan_rail_safe_path(start,goal):
    """Direct path first; otherwise use a bounded bidirectional RRT search."""
    start=np.asarray(start,dtype=float)
    goal=np.asarray(goal,dtype=float)
    if collision_free_edge(start,goal): return [start,goal]
    print('Direct motion intersects rail. Searching for an alternate arm posture...',flush=True)
    if not rail_clearance(start[0],start[1:])[0]:
        raise RuntimeError('Starting arm posture is inside rail clearance zone')
    if not rail_clearance(goal[0],goal[1:])[0]:
        raise RuntimeError('Destination arm posture is inside rail clearance zone')
    rng=np.random.default_rng(7)
    weight=np.array([2.0,1,1,1,.7,.7,.7])
    step=.28
    trees=[[(start,-1)],[(goal,-1)]]
    def nearest(tree,p):
        return min(range(len(tree)),key=lambda i: np.linalg.norm((tree[i][0]-p)*weight))
    def extend(tree,p):
        j=nearest(tree,p); old=tree[j][0]
        d=np.linalg.norm((p-old)*weight)
        if d<1e-8: return None
        nxt=old+(p-old)*min(1.,step/d)
        nxt[0]=np.clip(nxt[0],.15,3.20)
        nxt[1:]=np.clip(nxt[1:],-3.,3.)
        if not collision_free_edge(old,nxt): return None
        tree.append((nxt,j))
        return len(tree)-1
    def branch(tree,j):
        path=[]
        while j!=-1:
            state,parent=tree[j];path.append(state);j=parent
        return path[::-1]
    for k in range(3500):
        side=k%2
        tree=trees[side];other=trees[1-side]
        if rng.random()<.28: sample=other[-1][0]
        else:
            sample=np.r_[rng.uniform(.15,3.20),rng.uniform(-3.,3.,6)]
        j=extend(tree,sample)
        if j is None: continue
        candidate=tree[j][0]
        other_j=nearest(other,candidate)
        if collision_free_edge(other[other_j][0],candidate):
            first=branch(tree,j)
            second=branch(other,other_j)
            path=(first+second[::-1]) if side==0 else (second+first[::-1])
            # Greedy shortcut, keeping collision checks on every shortcut.
            i=0; shortened=[path[0]]
            while i<len(path)-1:
                nxt=len(path)-1
                while nxt>i+1 and not collision_free_edge(path[i],path[nxt]): nxt-=1
                shortened.append(path[nxt]);i=nxt
            print(f'Alternate path found ({len(shortened)-1} segments)',flush=True)
            return shortened
    raise RuntimeError('No collision-clear path found after planning; robot remains stopped.')

def jog_tick(jog, base_url, allowed):
    """Fetch jog commands from the control page, run one jog frame and
    publish the jog state back for the page to show."""
    import json
    from urllib.request import Request, urlopen
    try:
        with urlopen(base_url + '/jog_cmds', timeout=0.5) as r:
            commands = json.load(r)
    except Exception:
        commands = []
    for c in commands:
        if not allowed:
            jog.cancel()
            jog.fault = 'not available now - finish or reset the delivery first'
            continue
        t = c.get('type')
        if t == 'robot':
            jog.set_robot(c.get('idx', 0))
        elif t == 'step':
            jog.set_step(c.get('idx', 1))
        elif t == 'joint':
            jog.set_joint(int(c['j']), float(c['value']))
        elif t == 'rail':
            jog.set_rail(float(c['value']))
        elif t == 'cart':
            jog.cart_jog(int(c['axis']), int(c['sign']))
        elif t == 'home':
            jog.home()
    if allowed:
        jog.frame()
    elif jog.busy():
        jog.cancel()
        jog.fault = 'jog cancelled by safety stop'
    snap = jog.snapshot()
    snap['allowed'] = allowed
    try:
        urlopen(Request(base_url + '/jog_state', data=json.dumps(snap).encode(),
                        method='POST'), timeout=0.5).close()
    except Exception:
        pass


class TeachOnly(Exception):
    """Teach session: build the scene, skip the delivery."""


def main(colour='red', patient=2, teach=False):
    if colour not in layout.MEDICATION_SLOTS or patient not in TABLES:
        raise ValueError('Invalid medication colour or destination bed')
    safety=SafetyControl()
    env=swift.Swift()
    env.launch(realtime=True)
    containers=build_ward(env)
    equipment=SafetyEquipment(env, layout.KAWASAKI_BASE)
    safety.on_status = equipment.update

    # Compact Kawasaki medication-workstation light curtain.
    # Simulated optical guarding; not a physical safety device.
    from urllib.request import Request, urlopen
    import json

    yellow = [0.96, 0.75, 0.08, 1]
    dark = [0.12, 0.15, 0.18, 1]
    beam = [0.15, 0.75, 1.0, 0.20]

    def curtain_box(size, pos, color):
        obj = Cuboid(scale=size, pose=SE3(*pos), color=color)
        env.add(obj)
        return obj

    # Workstation centre: (1.65, -0.29).
    # The curtain is slightly wider than the tabletop.
    xmin, xmax = 0.94, 2.36
    ymin, ymax = -1.00, 0.39

    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]

    # Taller tabletop posts; transfer side deliberately has no beams.
    for cx, cy in corners:
        curtain_box(
            [0.055, 0.055, 0.91],
            (cx, cy, 1.225),
            yellow
        )
        curtain_box(
            [0.10, 0.10, 0.025],
            (cx, cy, 0.78),
            dark
        )

    # Three optical beam levels above the workstation.
    for i in (0, 1, 3):  # no beam on y=ymax, Doosan transfer side
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]

        for z in (0.90, 1.20, 1.52):
            curtain_box(
                [
                    max(abs(x2-x1), 0.012),
                    max(abs(y2-y1), 0.012),
                    0.006
                ],
                ((x1+x2)/2, (y1+y2)/2, z),
                beam
            )

    # Kawasaki light-curtain test actor.
    # Person stays outside the table and extends one arm into the beam.
    person_y = -0.40
    person_x = xmin - 0.95

    arm_length = 0.42
    arm_height = 0.94

    person_parts = [
        # Torso
        curtain_box(
            [0.24, 0.16, 0.48],
            (person_x, person_y, 0.75),
            [1.0, 0.35, 0.12, 1]
        ),
        # Legs
        curtain_box(
            [0.10, 0.11, 0.40],
            (person_x - 0.07, person_y, 0.30),
            dark
        ),
        curtain_box(
            [0.10, 0.11, 0.40],
            (person_x + 0.07, person_y, 0.30),
            dark
        ),
        # Upright head
        Cylinder(
            radius=0.12,
            length=0.22,
            pose=SE3(person_x, person_y, 1.10),
            color=[0.96, 0.72, 0.50, 1]
        ),
        # One simple arm reaching forward.
        curtain_box(
            [arm_length, 0.09, 0.09],
            (person_x + 0.12 + arm_length / 2,
             person_y, arm_height),
            [1.0, 0.35, 0.12, 1]
        ),
    ]

    env.add(person_parts[3])

    # Hide all parts until the curtain test is requested.
    for obj in person_parts:
        obj.T = SE3(person_x, person_y, -3)

    curtain_triggered = False

    def curtain_tick():
        nonlocal person_x, curtain_triggered

        if curtain_triggered:
            return

        try:
            with urlopen(safety.url, timeout=0.5) as response:
                status = json.load(response)
        except Exception:
            return

        if not status.get('intrusion_requested'):
            return

        # Walk toward the workcell with arm extended.
        # Stop the hand at the curtain boundary.
        hand_offset = 0.12 + arm_length
        person_x = min(
            xmin - hand_offset,
            person_x + 0.025
        )

        person_parts[0].T = SE3(person_x, person_y, 0.75)
        person_parts[1].T = SE3(person_x - 0.07, person_y, 0.30)
        person_parts[2].T = SE3(person_x + 0.07, person_y, 0.30)
        person_parts[3].T = SE3(person_x, person_y, 1.10)

        person_parts[4].T = SE3(
            person_x + 0.12 + arm_length / 2,
            person_y,
            arm_height
        )

        # Trigger exactly when the arm reaches the beam.
        if person_x + hand_offset >= xmin - 0.001:
            curtain_triggered = True
            print(
                '[LIGHT CURTAIN] Worker arm interrupted the beam.',
                flush=True
            )

            try:
                urlopen(
                    Request(
                        safety.url.replace(
                            '/status', '/curtain_breach'
                        ),
                        data=b'',
                        method='POST'
                    ),
                    timeout=0.5
                ).close()
            except Exception as exc:
                print(
                    '[LIGHT CURTAIN] Signal failed:',
                    exc,
                    flush=True
                )

    # Two rail-end approach scanners. A simulated worker advances from
    # outside either rail end and triggers only on crossing the defined
    # scanner threshold. This is illustrative 2D access monitoring.
    scanner_people = {}
    for side, sx, direction in (("left", -1.04, 1), ("right", 4.37, -1)):
        sy = 0.73
        parts = [
            curtain_box([0.23, 0.16, 0.46], (sx, sy, 0.73), [0.92, 0.38, 0.12, 1]),
            curtain_box([0.09, 0.10, 0.38], (sx-0.07, sy, 0.29), dark),
            curtain_box([0.09, 0.10, 0.38], (sx+0.07, sy, 0.29), dark),
            Cylinder(radius=0.11, length=0.21, pose=SE3(sx, sy, 1.08),
                     color=[0.96, 0.72, 0.50, 1]),
        ]
        env.add(parts[-1])
        # Keep inactive demonstration actors hidden until requested.
        for obj in parts:
            obj.T = SE3(sx, sy, -3)
        scanner_people[side] = {"x": sx, "start": sx, "direction": direction,
                                "y": sy, "parts": parts}
    scanner_triggered = False

    def scanner_tick():
        nonlocal scanner_triggered
        if scanner_triggered:
            return
        try:
            with urlopen(safety.url, timeout=0.5) as response:
                status = json.load(response)
        except Exception:
            return
        side = status.get('scanner_requested')
        if side not in scanner_people:
            return
        actor = scanner_people[side]
        actor['x'] += 0.028 * actor['direction']
        px, py = actor['x'], actor['y']
        for obj, dx, z in zip(actor['parts'], (0, -0.07, 0.07, 0),
                              (0.73, 0.29, 0.29, 1.08)):
            obj.T = SE3(px + dx, py, z)
        # Left and right scanner protective-field entry planes.
        crossed = (px + 0.12 >= -0.20) if side == 'left' else (px - 0.12 <= 3.62)
        if crossed:
            scanner_triggered = True
            print('[RAIL SCANNER] %s approach detected inside protective field.' % side.upper(), flush=True)
            try:
                urlopen(Request(safety.url.replace('/status', '/scanner_breach'),
                                data=b'', method='POST'), timeout=0.5).close()
            except Exception as exc:
                print('[RAIL SCANNER] Signal failed:', exc, flush=True)

    original_check = safety.check

    def checked_motion():
        curtain_tick()
        scanner_tick()
        original_check()

    safety.check = checked_motion

    robot=create_doosan()
    carriage=Cuboid(scale=[.38,.40,.12],pose=SE3(TRANSFER_X,RAIL_Y,CARRIAGE_Z),color=[.2,.25,.32,1])
    env.add(carriage)
    groups=[]
    for filenames in MESH_FILES:
        group=[]
        for filename in filenames:
            mesh=Mesh(filename=str(MESH_DIR/filename),scale=[.001]*3)
            mesh.filename=swift_mesh_path(mesh.filename)
            env.add(mesh)
            group.append(mesh)
        groups.append(group)
    # Bottle is selected from the live RGB-D detection before pickup.
    bottle = None
    x=TRANSFER_X
    q=np.zeros(6)
    holding=False
    grasp=SE3()
    def redraw():
        robot.base=base_pose(x)
        robot.q=q
        carriage.T=SE3(x,RAIL_Y,CARRIAGE_Z)
        T=base_pose(x)
        for mesh in groups[0]: mesh.T=T
        for i in range(6):
            T=T*SE3.Rt(rpy2r(JOINT_RPY[i],order='zyx'),JOINT_XYZ[i])*SE3.Rz(q[i])
            for mesh in groups[i+1]: mesh.T=T
        if holding:
            # Follow the same grasp trajectory as V2 but keep the cylinder upright.
            world_grasp = robot.fkine(q) * grasp
            world_matrix = np.asarray(world_grasp, dtype=float)
            position = world_matrix[:3, 3]
            bottle.T = SE3(*[float(v) for v in position])
    def move(new_x,new_q,seconds):
        nonlocal x,q
        start=np.r_[x,q]
        target=np.r_[new_x,np.asarray(new_q,dtype=float)]
        path=plan_rail_safe_path(start,target)
        lengths=[np.linalg.norm((b-a)*np.array([2,1,1,1,.7,.7,.7])) for a,b in zip(path[:-1],path[1:])]
        total=max(sum(lengths),1e-9)
        for (a,b),distance in zip(zip(path[:-1],path[1:]),lengths):
            duration=max(.4,seconds*distance/total)
            n=max(2,int(duration/.05))
            for i in range(n+1):
                safety.check()
                u=i/n;s=10*u**3-15*u**4+6*u**5
                state=a+(b-a)*s
                x=float(state[0]);q=state[1:].copy()
                redraw();env.step(.05);time.sleep(.05)
    kawasaki = None
    try:
        redraw(); env.step(.1)
        # Stage 1: select the medication from RGB-D observations.
        print("\nSTAGE 1 — Kawasaki picking %s medication for bed %d" % (colour.upper(), patient), flush=True)
        try:
            kawasaki = KawasakiCell(env)
            kawasaki.safety = safety
            camera = RGBDCamera(env)
            sensor = PhotoelectricSensor(env)
            if teach:
                raise TeachOnly()
            kawasaki.pause(0.3)
            pinhole = PinholeCamera(camera.T)
            rgbd = SimulatedRGBD(pinhole, containers)
            rgb, depth = rgbd.capture()
            detected = detect_containers(rgb, depth, pinhole)
            shelf = [d for d in detected if d.position[1] < -0.55]
            capture_path = Path(__file__).resolve().parent.parent / 'captures' / 'latest_detection.png'
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            save_debug_image(str(capture_path), rgb, depth, shelf)
            choices = [d for d in shelf if d.colour == colour]
            if not choices:
                raise RuntimeError('RGB-D camera did not detect %s medication on shelf' % colour)
            detection = choices[0]
            bottle = find_object_at(containers, detection.position)
            if bottle is None:
                raise RuntimeError('Detected medication does not match a graspable container')
            print('[RGB-D] Detected %s medication at %s' % (colour, np.round(detection.position, 3)), flush=True)
            kawasaki.pick_and_place(bottle, SE3(detection.position),
                                    layout.TRANSFER_POINT, camera=camera, sensor=sensor)
            # Keep Kawasaki in its home position as Doosan takes over.
            bottle.T = SE3(TRANSFER_X, TRANSFER_Y, TRANSFER_Z + BOTTLE_H / 2)
            print("STAGE 2 — Doosan collecting %s medication for bed %d" % (colour.upper(), patient), flush=True)
        except (SimulationStopped, TeachOnly):
            if not teach:
                print("[SAFETY] Kawasaki task cancelled; Doosan will not start.", flush=True)
            raise
        except Exception as exc:
            print("Kawasaki failed; Doosan will NOT start a fake handoff:", repr(exc), flush=True)
            raise
        print('\nMediAssist — rail collision-aware path planning',flush=True)
        try:
            tx,ty,tz=TABLES[patient]
            drop_x=float(np.clip(tx,.15,3.20))
            pickup_z=TRANSFER_Z+BOTTLE_H
            drop_z=tz+BOTTLE_H
            print('Solving rail-clear pickup/drop configurations...',flush=True)
            qa=solve_ik(robot,TRANSFER_X,SE3(TRANSFER_X,TRANSFER_Y,pickup_z+.15),q)
            qb=solve_ik(robot,TRANSFER_X,SE3(TRANSFER_X,TRANSFER_Y,pickup_z),qa)
            qc=solve_ik(robot,drop_x,SE3(tx,ty,drop_z+.15),qb)
            qd=solve_ik(robot,drop_x,SE3(tx,ty,drop_z),qc)
            print('Checking motion and delivering...',flush=True)
            move(TRANSFER_X,qa,2)
            move(TRANSFER_X,qb,1.5)
            robot.base=base_pose(x); robot.q=q
            grasp=robot.fkine(q).inv()*bottle.T
            holding=True
            # The container has been grasped and is leaving the transfer plate.
            # Return the simulated photoelectric presence indicator to RED (empty).
            sensor.set_detected(False)
            env.step(.05)
            move(TRANSFER_X,qa,1.5)
            # Calculate where the bottle will be, including its grasp offset.
            desired_pos = np.array([tx, ty, tz + BOTTLE_H / 2], dtype=float)
            grasp_matrix = np.asarray(grasp, dtype=float)

            robot.base = base_pose(drop_x)
            target_q = qd.copy()

            # Correct the arm target, rather than moving the bottle separately.
            for _ in range(5):
                ee = np.asarray(robot.fkine(target_q), dtype=float)
                predicted = (ee @ grasp_matrix)[:3, 3]
                error = desired_pos - predicted

                if np.linalg.norm(error) < .012:
                    break

                corrected_ee = ee[:3, 3] + error
                target_q = solve_ik(
                    robot, drop_x, SE3(*corrected_ee), target_q
                )

            ee = np.asarray(robot.fkine(target_q), dtype=float)
            predicted = (ee @ grasp_matrix)[:3, 3]
            error_mm = float(np.linalg.norm(desired_pos - predicted) * 1000)

            print(
                '[PLACEMENT] Predicted bottle offset: %.1f mm' % error_mm,
                flush=True
            )

            if error_mm > 20:
                raise RuntimeError(
                    'Bottle placement could not be aligned within 20 mm.'
                )

            # Carry bottle to table using actual robot movement.
            move(drop_x,qc,3)
            move(drop_x,target_q,1.5)

            # Detach bottle at its actual held position.
            safety.check()
            holding=False
            print('Medication released on bedside table.',flush=True)

            # Original working retreat.
            move(drop_x,qc,1.5)
            print('Delivery completed.',flush=True)
        except SimulationStopped:
            print("[SAFETY] Doosan task cancelled.", flush=True)
            raise
        except RuntimeError as exc:
            print("[PLANNING]", str(exc), flush=True)
            raise
        print('Delivery complete. Awaiting next operator command.', flush=True)

    except TeachOnly:
        print('[TEACH] Scene ready for jogging.', flush=True)
    except SimulationStopped:
        print('[SAFETY] Motion cancelled. Swift remains OPEN and stationary.', flush=True)
    except Exception as exc:
        print('[FAULT] Task stopped:', repr(exc), flush=True)
    else:
        print('[SAFETY] Delivery sequence ended; awaiting operator.', flush=True)
        safety.complete()
    # No further robot trajectory commands. Preserve the scene during STOP and RESET.
    # Only a separate START requests worker shutdown and a fresh initial scene.
    print('[SAFETY] Holding current robot poses until START NEW DELIVERY.', flush=True)

    # Teach / jog: allowed once the delivery is COMPLETE or in a TEACH
    # session, never while stopped. One jog frame per loop - nothing blocks.
    jog = None
    if kawasaki is not None:
        from tasks.jog_panel import JogController

        class DoosanHandle:
            """Gives the jog controller access to this worker's Doosan."""
            def __init__(self):
                self.robot = robot
            @property
            def x(self):
                return x
            @x.setter
            def x(self, value):
                set_doosan(value, q)
            @property
            def q(self):
                return q
            @q.setter
            def q(self, value):
                set_doosan(x, value)
            def redraw(self):
                redraw()
            def park_q(self):
                if not hasattr(self, '_park'):
                    self._park = solve_ik(robot, TRANSFER_X,
                                          SE3(TRANSFER_X, 0.45, 1.25), np.zeros(6))
                return self._park.copy()

        def set_doosan(new_x, new_q):
            nonlocal x, q
            x = float(new_x)
            q = np.asarray(new_q, dtype=float)

        jog = JogController(kawasaki, DoosanHandle())
        jog.set_mode(True)
    jog_url = safety.url.replace('/status', '')
    try:
        while True:
            state = safety.status()
            if state == 'RESTARTING':
                print('[SAFETY] New task authorised. Closing previous scene.', flush=True)
                break
            if jog is not None:
                jog_tick(jog, jog_url, state in ('COMPLETE', 'TEACH'))
            env.step(.05)
            time.sleep(.05)
    finally:
        safety.close()
        env.close()

if __name__ == '__main__':
    import sys
    if '--worker' not in sys.argv:
        from tasks.safety_control import SafetyLauncher
        SafetyLauncher().run()
    else:
        try:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--worker', action='store_true')
            parser.add_argument('--colour', choices=('red','green','blue'), default='red')
            parser.add_argument('--bed', type=int, choices=(1,2,3), default=2)
            parser.add_argument('--teach', action='store_true')
            args = parser.parse_args()
            main(colour=args.colour, patient=args.bed, teach=args.teach)
        except SimulationStopped as exc:
            print('[SAFETY] Worker stopped:', exc, flush=True)
        except KeyboardInterrupt:
            print('[SAFETY] Worker interrupted.', flush=True)
