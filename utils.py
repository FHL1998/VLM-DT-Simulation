def calculate_world_bbox(obj_data):
    """
    Calculate an object's bounding box in world space coordinates considering rotation
    Returns [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    if not obj_data or "bound_box" not in obj_data or "matrix_world" not in obj_data:
        return None

    # Get the bound_box vertices
    bound_box = obj_data["bound_box"]

    # Get the world transformation matrix
    matrix_world = obj_data["matrix_world"]

    # Convert the matrix to a usable format for transformation
    # The matrix_world is a 4x4 transformation matrix that includes translation, rotation and scale

    # Function to transform a point using the matrix
    def transform_point(point, matrix):
        # Create homogeneous coordinates (add a 1 at the end)
        homogeneous_point = [point[0], point[1], point[2], 1.0]

        # Apply transformation
        transformed = [0, 0, 0, 0]
        for i in range(4):
            for j in range(4):
                transformed[i] += matrix[i][j] * homogeneous_point[j]

        # Return 3D coordinates
        return [transformed[0], transformed[1], transformed[2]]

    # Transform all bounding box vertices to world space
    world_points = []
    for point in bound_box:
        world_point = transform_point(point, matrix_world)
        world_points.append(world_point)

    # Find min and max for each axis
    min_x = min(point[0] for point in world_points)
    min_y = min(point[1] for point in world_points)
    min_z = min(point[2] for point in world_points)
    max_x = max(point[0] for point in world_points)
    max_y = max(point[1] for point in world_points)
    max_z = max(point[2] for point in world_points)

    return [min_x, min_y, min_z, max_x, max_y, max_z]


def get_object_data(scene_info, obj_name):
    """Find object data in the Blender scene JSON"""
    # This assumes scene_info is available in the scope
    for name, data in scene_info["objects"].items():
        if obj_name.lower() in name.lower():
            return data
    return None