# MediAssist System Design

## System Overview

MediAssist is an aged-care medication retrieval and delivery system using two robotic arms working within a shared environment.

## Robot 1 - Kawasaki RS007N

Robot 1 operates at the medication storage station. It will identify the requested colour-coded dummy medication container, pick it from the storage area and place it at the transfer point.

## Robot 2 - Doosan M0609

Robot 2 collects the medication container from the transfer point and delivers it to the selected bedside location. The robot is planned to operate on a simulated linear rail so that it can access multiple bedside locations.

## Planned Sequence

1. User selects a medication and destination bed.
2. Kawasaki RS007N retrieves the requested container.
3. Kawasaki places the container at the transfer point.
4. Doosan M0609 collects the container.
5. Doosan moves to the selected bed location.
6. Doosan places the container at the bedside collection point.
7. System returns to a safe state ready for the next request.

## Safety Features

The system will incorporate emergency-stop behaviour, collision detection, safety zones and controlled reset/restart behaviour. Additional sensing may be incorporated for medication detection and confirmation of object transfer.
