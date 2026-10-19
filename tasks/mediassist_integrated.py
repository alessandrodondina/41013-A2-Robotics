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
from tasks.kawasaki_pick_place import KawasakiCell
from environment.sensors import RGBDCamera, PhotoelectricSensor
from robot_models.doosan_m0609 import create_doosan, JOINT_XYZ, JOINT_RPY

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

def rail_clearance(x, joints):
    """Conservative approximate link-segment vs rail-body test.
    Rail body occupies x=-.05..3.45, y=.61..89, z=.65..83.
    Added 35mm clearance. Ignore anchored base-to-shoulder segment.
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

def main():
    env=swift.Swift()
    env.launch(realtime=True)
    containers=build_ward(env)
    robot=create_doosan()
    carriage=Cuboid(scale=[.38,.40,.12],pose=SE3(TRANSFER_X,RAIL_Y,CARRIAGE_Z),color=[.2,.25,.32,1])
    env.add(carriage)
    groups=[]
    for filenames in MESH_FILES:
        group=[]
        for filename in filenames:
            mesh=Mesh(filename=str(MESH_DIR/filename),scale=[.001]*3)
            env.add(mesh)
            group.append(mesh)
        groups.append(group)
    # Share the actual red shelf container with both robots.
    red_shelf_index = layout.SHELF_ORDER.index("red")
    bottle = containers[red_shelf_index]
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
                u=i/n;s=10*u**3-15*u**4+6*u**5
                state=a+(b-a)*s
                x=float(state[0]);q=state[1:].copy()
                redraw();env.step(.05);time.sleep(.05)
    redraw(); env.step(.1)
    # Stage 1: Kawasaki moves the real red cylinder to the transfer plate.
    print("\nSTAGE 1 — Kawasaki picking RED medication", flush=True)
    try:
        kawasaki = KawasakiCell(env)
        camera = RGBDCamera(env)
        sensor = PhotoelectricSensor(env)
        kawasaki.pause(1.0)
        kawasaki.pick_and_place(bottle, layout.MEDICATION_SLOTS["red"][0],
                                layout.TRANSFER_POINT, camera=camera, sensor=sensor)
        # Keep Kawasaki in its home position as Doosan takes over.
        bottle.T = SE3(TRANSFER_X, TRANSFER_Y, TRANSFER_Z + BOTTLE_H / 2)
        print("STAGE 2 — Doosan collecting RED medication", flush=True)
    except Exception as exc:
        print("Kawasaki failed; Doosan will NOT start a fake handoff:", repr(exc), flush=True)
        raise
    print('\nMediAssist — rail collision-aware path planning',flush=True)
    try:
        patient=2
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
        move(drop_x,qc,3)
        move(drop_x,qd,1.5)
        # Visual placement correction: V2's end-effector reaches its target,
        # but its unconstrained wrist can leave the bottle offset from the table.
        # Blend the upright bottle to the table over 1.5 s, without snapping.
        # Keep the arm stationary during this virtual handoff; all arm moves
        # remain subject to V2's collision-aware planner.
        start_pos=np.asarray(bottle.T,dtype=float)[:3,3].copy()
        end_pos=np.array([tx,ty,tz+BOTTLE_H/2],dtype=float)
        print('Final tabletop placement offset:',round(float(np.linalg.norm(end_pos-start_pos)),3),'m',flush=True)
        # Visual final placement: hold the arm stationary, move the upright
        # bottle to the exact tabletop height, then release the virtual grasp.
        # This is a visual correction, not a physically modelled gripper.
        for k in range(61):
            u=k/60
            s=10*u**3-15*u**4+6*u**5
            p=start_pos+(end_pos-start_pos)*s
            bottle.T=SE3(*[float(v) for v in p])
            env.step(.05)
            time.sleep(.05)
        # Settle on the table before releasing the virtual grasp.
        for _ in range(12):
            env.step(.05)
            time.sleep(.05)
        holding=False
        print('Medication released on bedside table.',flush=True)
        move(drop_x,qc,1.5)
        print('Delivery completed.',flush=True)
    except RuntimeError as exc:
        print('\n'+str(exc),flush=True)
        print('Planning stopped before the rejected segment; working backup is unchanged.',flush=True)
    print('Swift stays open. Ctrl+C to quit.',flush=True)
    try:
        while True:
            env.step(.05); time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__=='__main__':
    main()
