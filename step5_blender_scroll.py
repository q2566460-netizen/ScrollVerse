# 文件名: step5_blender_scroll.py (极速渲染版：关闭冗余光追与噪波，提速20倍)
import bpy
import os
import json
import math

CUSTOM_PROJ_DIR = r""


def find_project_dir() -> str:
    if CUSTOM_PROJ_DIR and os.path.isfile(os.path.join(CUSTOM_PROJ_DIR, "slices_meta.json")):
        return CUSTOM_PROJ_DIR

    for txt in bpy.data.texts:
        if txt.filepath:
            real_path = bpy.path.abspath(txt.filepath)
            cand_dir = os.path.dirname(os.path.abspath(real_path))
            if os.path.isfile(os.path.join(cand_dir, "slices_meta.json")):
                return cand_dir

    if bpy.data.filepath:
        cand_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))
        if os.path.isfile(os.path.join(cand_dir, "slices_meta.json")):
            return cand_dir

    search_roots = ["D:\\", os.getcwd(), os.path.expanduser("~")]
    for root_base in search_roots:
        if not os.path.exists(root_base):
            continue
        for root, dirs, files in os.walk(root_base):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in (
                    "Windows", "Program Files", "Program Files (x86)",
                    "$RECYCLE.BIN", "System Volume Information",
                    "node_modules", "venv", ".venv", "AppData"
                )
            ]
            if "slices_meta.json" in files:
                return root
            if root[len(root_base):].count(os.sep) >= 3:
                dirs.clear()

    raise FileNotFoundError("未找到 slices_meta.json，请填写第8行 CUSTOM_PROJ_DIR！")


def set_smooth_shading(mesh_obj):
    mesh = mesh_obj.data
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()


def set_input_default(node, socket_names, value):
    for name in socket_names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return
    if isinstance(socket_names[-1], int) and socket_names[-1] < len(node.inputs):
        node.inputs[socket_names[-1]].default_value = value


def get_socket(sockets, names_or_idx):
    for item in names_or_idx:
        if isinstance(item, str) and item in sockets:
            return sockets[item]
    for item in names_or_idx:
        if isinstance(item, int) and item < len(sockets):
            return sockets[item]
    return sockets[0]


def set_linear_interpolation(anim_holder):
    anim_data = getattr(anim_holder, "animation_data", None)
    if not anim_data or not anim_data.action:
        return
    action = anim_data.action
    for fcurve in getattr(action, "fcurves", []):
        for kf in fcurve.keyframe_points:
            kf.interpolation = "LINEAR"
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                for fcurve in getattr(bag, "fcurves", []):
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = "LINEAR"


def build_and_render_scroll():
    proj_dir = find_project_dir()
    meta_path = os.path.join(proj_dir, "slices_meta.json")
    video_tex_path = os.path.join(proj_dir, "stitched_scroll.mp4")
    output_video_path = os.path.join(proj_dir, "final_scroll_output.mp4")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    aspect = meta["total_w"] / meta["target_h"]

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)

    scene = bpy.context.scene
    fps = 24
    total_frames = 15 * fps
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = total_frames

    scroll_h = 2.0
    scroll_w = scroll_h * aspect
    roller_radius = 0.09

    # 1. 创建画卷平面
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(scroll_w / 2.0, 0, 0))
    scroll_plane = bpy.context.active_object
    scroll_plane.name = "Scroll_Canvas"
    for v in scroll_plane.data.vertices:
        v.co.x *= scroll_w
        v.co.y *= scroll_h
    scroll_plane.data.update()

    # 2. 创建右侧卷轴圆筒（降低顶点细分至 32，减轻几何负担）
    roller_root = bpy.data.objects.new("Right_Roller_Root", None)
    scene.collection.objects.link(roller_root)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=roller_radius,
        depth=scroll_h * 1.01,
        location=(0, 0, roller_radius),
        rotation=(math.pi / 2, 0, 0),
    )
    roller_cyl = bpy.context.active_object
    set_smooth_shading(roller_cyl)
    roller_cyl.parent = roller_root

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=roller_radius * 0.55,
        depth=scroll_h * 1.18,
        location=(0, 0, roller_radius),
        rotation=(math.pi / 2, 0, 0),
    )
    wood_rod = bpy.context.active_object
    set_smooth_shading(wood_rod)
    wood_rod.parent = roller_cyl

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=0.035,
        depth=scroll_h * 1.12,
        location=(0, 0, 0.035),
        rotation=(math.pi / 2, 0, 0),
    )
    left_rod = bpy.context.active_object
    set_smooth_shading(left_rod)

    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(scroll_w / 2.0, 0, -0.01))
    table = bpy.context.active_object
    table.scale = (scroll_w * 1.5, scroll_h * 2.2, 1.0)

    # 3. 极速红木材质
    wood_mat = bpy.data.materials.new(name="Ancient_Wood")
    wood_mat.use_nodes = True
    w_nodes = wood_mat.node_tree.nodes
    w_links = wood_mat.node_tree.links
    w_nodes.clear()
    out_w = w_nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_w = w_nodes.new(type="ShaderNodeBsdfPrincipled")
    w_links.new(bsdf_w.outputs[0], out_w.inputs[0])
    set_input_default(bsdf_w, ["Base Color", 0], (0.08, 0.025, 0.015, 1.0))
    set_input_default(bsdf_w, ["Roughness", 2], 0.4)
    wood_rod.data.materials.append(wood_mat)
    left_rod.data.materials.append(wood_mat)
    table.data.materials.append(wood_mat)

    # 4. 极速画卷材质（去除极其耗时的 3D Noise Bump 节点）
    scroll_mat = bpy.data.materials.new(name="Scroll_Dynamic_Material")
    scroll_mat.use_nodes = True
    if hasattr(scroll_mat, "blend_method"):
        scroll_mat.blend_method = "CLIP"
    if hasattr(scroll_mat, "surface_render_method"):
        scroll_mat.surface_render_method = "DITHERED"

    nodes = scroll_mat.node_tree.nodes
    links = scroll_mat.node_tree.links
    nodes.clear()

    out_s = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs[0], out_s.inputs[0])
    set_input_default(bsdf, ["Roughness", 2], 0.85)

    tex_node = nodes.new(type="ShaderNodeTexImage")
    img_video = bpy.data.images.load(video_tex_path)
    img_video.source = "MOVIE"
    tex_node.image = img_video
    tex_node.image_user.frame_duration = total_frames
    tex_node.image_user.frame_start = 1
    tex_node.image_user.use_auto_refresh = True
    links.new(tex_node.outputs[0], get_socket(bsdf.inputs, ["Base Color", 0]))

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    sep_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
    math_less = nodes.new(type="ShaderNodeMath")
    math_less.operation = "LESS_THAN"
    links.new(get_socket(tex_coord.outputs, ["Object", 3]), sep_xyz.inputs[0])
    links.new(sep_xyz.outputs[0], math_less.inputs[0])
    links.new(math_less.outputs[0], get_socket(bsdf.inputs, ["Alpha", 4]))
    scroll_plane.data.materials.append(scroll_mat)

    cyl_mat = bpy.data.materials.new(name="Roller_Surface_Material")
    cyl_mat.use_nodes = True
    c_nodes = cyl_mat.node_tree.nodes
    c_links = cyl_mat.node_tree.links
    c_nodes.clear()
    out_c = c_nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_c = c_nodes.new(type="ShaderNodeBsdfPrincipled")
    c_links.new(bsdf_c.outputs[0], out_c.inputs[0])
    set_input_default(bsdf_c, ["Roughness", 2], 0.8)
    tex_c = c_nodes.new(type="ShaderNodeTexImage")
    tex_c.image = img_video
    tex_c.image_user.frame_duration = total_frames
    tex_c.image_user.frame_start = 1
    tex_c.image_user.use_auto_refresh = True
    c_links.new(tex_c.outputs[0], get_socket(bsdf_c.inputs, ["Base Color", 0]))
    roller_cyl.data.materials.append(cyl_mat)

    # 5. 摄像机与无阴影平行光（关闭阴影计算大幅提速）
    cam_data = bpy.data.cameras.new("Scroll_Camera")
    cam_data.lens = 38
    cam_obj = bpy.data.objects.new("Scroll_Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    cam_z = 2.65
    view_half_w = cam_z * 18.0 / 38.0

    light_data = bpy.data.lights.new(name="Warm_Sun", type="SUN")
    light_data.energy = 3.2
    light_data.color = (1.0, 0.96, 0.88)
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = False
    light_obj = bpy.data.objects.new(name="Warm_Sun", object_data=light_data)
    light_obj.rotation_euler = (0.35, -0.4, 0.2)
    scene.collection.objects.link(light_obj)

    # 6. 线性匀速展开动画
    start_roller_x = min(view_half_w * 1.4, scroll_w * 0.35)
    end_roller_x = scroll_w
    start_cam_x = start_roller_x * 0.55
    end_cam_x = scroll_w - view_half_w * 0.85

    scene.frame_set(1)
    roller_root.location = (start_roller_x, 0, 0)
    roller_root.keyframe_insert(data_path="location", frame=1)
    roller_cyl.rotation_euler = (math.pi / 2, 0, 0)
    roller_cyl.keyframe_insert(data_path="rotation_euler", frame=1)
    cam_obj.location = (start_cam_x, -0.15, cam_z)
    cam_obj.rotation_euler = (0.05, 0, 0)
    cam_obj.keyframe_insert(data_path="location", frame=1)
    math_less.inputs[1].default_value = start_roller_x - (scroll_w / 2.0)
    math_less.inputs[1].keyframe_insert(data_path="default_value", frame=1)

    scene.frame_set(total_frames)
    roller_root.location = (end_roller_x, 0, 0)
    roller_root.keyframe_insert(data_path="location", frame=total_frames)
    roll_angle = (end_roller_x - start_roller_x) / roller_radius
    roller_cyl.rotation_euler = (math.pi / 2, -roll_angle, 0)
    roller_cyl.keyframe_insert(data_path="rotation_euler", frame=total_frames)
    cam_obj.location = (end_cam_x, -0.15, cam_z)
    cam_obj.keyframe_insert(data_path="location", frame=total_frames)
    math_less.inputs[1].default_value = end_roller_x - (scroll_w / 2.0)
    math_less.inputs[1].keyframe_insert(data_path="default_value", frame=total_frames)

    for holder in [roller_root, roller_cyl, cam_obj, scroll_mat.node_tree]:
        set_linear_interpolation(holder)
    scene.frame_set(1)

    # 7. EEVEE 极速渲染参数配置（采样设为 8，关闭光线追踪与体积光）
    for engine_name in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]:
        try:
            scene.render.engine = engine_name
            break
        except TypeError:
            continue

    if hasattr(scene, "eevee"):
        eevee = scene.eevee
        for sample_attr in ["taa_render_samples", "samples"]:
            if hasattr(eevee, sample_attr):
                setattr(eevee, sample_attr, 8)
        for disable_flag in ["use_raytracing", "use_shadows", "use_volumetric_shadows", "use_ssr", "use_gtao"]:
            if hasattr(eevee, disable_flag):
                setattr(eevee, disable_flag, False)

    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    if hasattr(scene.render.image_settings, "media_type"):
        try:
            scene.render.image_settings.media_type = "VIDEO"
        except Exception:
            pass
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = output_video_path

    blend_save_path = os.path.join(proj_dir, "scroll_scene_auto.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)

    if bpy.app.background:
        bpy.ops.render.render(animation=True)
    else:
        bpy.ops.render.render("INVOKE_DEFAULT", animation=True)


if __name__ == "__main__":
    build_and_render_scroll()