# Constructing Reliable System for Autonomous Learning
## Extracting Information from Demonstration
### Initial State
Extract
```bash
python reset/extract_from_demo/extract_initial_state.py --data-dir dataset/scripted/cabinet --output-dir reset/extracted_info
```

Visualization
```bash
python reset/visualization/visualize_initial_state.py \
    --dataset dataset/scripted/cabinet \
    --use-viser
```

Usage
```python
from reset.scripts.get_info import get_initial_state
initial_state = get_initial_state()
```
Example usage in `reset/scripts/test_initial_states.py`
```bash
python reset/scripts/test_initial_state.py --furniture cabinet
```

### Grasp

Extract
```bash
python -m reset.extract_from_demo.extract_grasp --grasps reset/extracted_info/grasps.json --output reset/extracted_info
```
Visualization
```bash
python reset/visualization/visualize_extracted_grasp.py \
    --summary reset/extracted_info/grasp_summary.pkl \
    --use-viser
```
Usage
```python
from reset.scripts.get_info import get_grasp_eef_pose
ee_pose = get_grasp_eef_pose(
            part_name="cabinet_body",
            pose=part_pose,
            furniture_name="cabinet",
            mode=0
        )
```
Example usage in `reset/scripts/test_grasp.py`
```bash
python reset/scripts/test_grasp.py --part-name cabinet_body --furniture cabinet --mode 1
```

### Object Affordance
Extract
```bash
python -m reset.extract_from_demo.extract_object_affordance --dataset dataset/scripted/cabinet/2025-10-30-23:41:04 --output reset/extracted_info 
```

Visualization
```bash
python reset/visualization/visualize_object_affordance.py \
  --json reset/extracted_info/object_affordance_trajectories.json \
  --furniture cabinet \
  --pair cabinet_top cabinet_body \
  --sample 0 \
  --fps 30 \
  --use-viser
```

Usage
```python
from reset.scripts.get_info import get_object_affordance
trajectory = get_object_affordance(
            base_part="cabinet_body,
            target_part="cabinet_door_left,
            furniture_name="cabinet",
            mode=0
        )
```
Example usage in `reset/scripts/test_object_affordance.py`
```bash
python reset/scripts/test_object_affordance.py --base-part cabinet_body --target-part cabinet_door_left --furniture cabinet --mode 0
```