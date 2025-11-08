# Constructing Reliable System for Autonomous Learning
## Extracting Information from Demonstration
### Extraction
```bash
python reset/scripts/extract.py \
    --dataset dataset/scripted/cabinet/2025-10-30-23:41:04/2025-10-30-23:41:04.pkl \
    --furniture cabinet \
    --contact-tolerance 0.012 \
    --output reset/extracted_info
```
By running above code, it will save two files. `distances.pkl` and `extracted_info.pkl`. You will be using the `extracted_info.pkl` which has the format as below.
```json
{
  "furniture": "furniture_name",
  "num_actions": N,
  "actions": [
    {
      "type": "MOVE",
      "grasped_part": "part_name",
      "start_step": 0,
      "end_step": 100,
      "initial_pose": [x, y, z, qx, qy, qz, qw],
      "end_pose": [x, y, z, qx, qy, qz, qw],
      "grasp_pose": {
        "relative_position": [x, y, z],
        "relative_quaternion": [qx, qy, qz, qw]
      }
    },
    {
      "type": "INTERACT",
      "target_part": "part_name",
      "base_part": "part_name",
      "start_step": 100,
      "end_step": 200,
      "target_initial_pose": [x, y, z, qx, qy, qz, qw],
      "base_initial_pose": [x, y, z, qx, qy, qz, qw],
      "target_end_pose": [x, y, z, qx, qy, qz, qw],
      "base_end_pose": [x, y, z, qx, qy, qz, qw],
      "grasp_pose": {
        "relative_position": [x, y, z],
        "relative_quaternion": [qx, qy, qz, qw]
      },
      "affordance_trajectory": [
        [x, y, z, qx, qy, qz, qw],  // relative pose at each frame
        ...
      ]
    }
  ]
}
```
`distances.pkl` is a file that saves all part-part and part-gripper distance. This is saved for the convenience of information extraction  (just in case). You won't be using this for later planning.

### Visualization
```bash
python reset/visualization/visualize_extracted_info.py \
  --extraction-file path/to/extracted_info.pkl \
  --dataset path/to/dataset.pkl \
  --use-viser
```