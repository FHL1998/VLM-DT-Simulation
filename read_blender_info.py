import bpy
import sys
import os


def read_blender_scene(blend_file_path):
    """
    Opens a Blender file and extracts information about objects in the scene.

    Args:
        blend_file_path (str): Path to the .blend file

    Returns:
        dict: Dictionary containing scene information
    """
    # Check if file exists
    if not os.path.exists(blend_file_path):
        print(f"Error: File {blend_file_path} not found.")
        return None

    # Load the .blend file
    try:
        bpy.ops.wm.open_mainfile(filepath=blend_file_path)
        print(f"Successfully opened {blend_file_path}")
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

    # Get scene information
    scene_info = {
        "scene_name": bpy.context.scene.name,
        "objects": {}
    }

    # Iterate through all objects in the scene
    for obj in bpy.context.scene.objects:
        # Get basic object information
        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "visible": obj.visible_get(),

            # Location (position)
            "location": {
                "x": obj.location.x,
                "y": obj.location.y,
                "z": obj.location.z
            },

            # Rotation (in radians)
            "rotation_euler": {
                "x": obj.rotation_euler.x,
                "y": obj.rotation_euler.y,
                "z": obj.rotation_euler.z
            },

            # Scale
            "scale": {
                "x": obj.scale.x,
                "y": obj.scale.y,
                "z": obj.scale.z
            },

            # Dimensions
            "dimensions": {
                "x": obj.dimensions.x,
                "y": obj.dimensions.y,
                "z": obj.dimensions.z
            },

            # Bounding box
            "bound_box": [[v[0], v[1], v[2]] for v in obj.bound_box],

            # Object matrix (transformation matrix)
            "matrix_world": [[v for v in row] for row in obj.matrix_world],
        }

        # Add object-type specific information
        if obj.type == 'MESH':
            obj_info.update({
                "vertex_count": len(obj.data.vertices),
                "polygon_count": len(obj.data.polygons),
                "material_count": len(obj.material_slots)
            })

        # Add to the scene information
        scene_info["objects"][obj.name] = obj_info

    return scene_info


def save_scene_info_to_json(scene_info, output_path):
    """Save scene information to a JSON file"""
    import json
    with open(output_path, 'w') as f:
        json.dump(scene_info, f, indent=4)
    print(f"Scene information saved to {output_path}")


if __name__ == "__main__":
    blend_file_path = "assets/scene.blend"
    if len(sys.argv) > 1 and '--' in sys.argv:
        args = sys.argv[sys.argv.index('--') + 1:]
        if args:
            blend_file_path = args[0]
            scene_info = read_blender_scene(blend_file_path)

            # Default output file is next to the input file
            output_path = os.path.splitext(blend_file_path)[0] + "_info.json"
            if len(args) > 1:
                output_path = args[1]

            if scene_info:
                save_scene_info_to_json(scene_info, output_path)
    else:
        # Use the provided path from your script
        scene_info = read_blender_scene(blend_file_path)
        if scene_info:
            # Print summary
            print(f"Scene: {scene_info['scene_name']}")
            print(f"Objects found: {len(scene_info['objects'])}")
            for name, obj in scene_info['objects'].items():
                print(f"  - {name} ({obj['type']}): Position {obj['location']}, Scale {obj['scale']}")

            output_path = os.path.splitext(blend_file_path)[0] + "_info.json"
            save_scene_info_to_json(scene_info, output_path)