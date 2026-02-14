import os
import numpy as np
from PIL import Image
import moderngl
import shutil
from tqdm import tqdm

class FilmicLightRenderer:

    def __init__(self,input_path,mask_path,albedo_path,normal_path,output_base_path,copy_source=True):
        self.copy_source = copy_source
        self.ctx=None
        try:
            self.ctx = moderngl.create_standalone_context()
            print("OpenGL context created successfully.")
        except Exception as e:
            print(f"Failed to create OpenGL context: {e}")
            return
        self.window_size = (512, 512) #默认窗口大小，后续会根据输入图片尺寸进行修改
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;
                void main() { gl_Position = vec4(in_position, 1.0); uv = in_texcoord_0;}
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D tex_input;
                uniform sampler2D tex_mask;
                uniform sampler2D tex_albedo;
                uniform sampler2D tex_normal;
                uniform vec3 light_pos;
                uniform vec3 light_color;
                uniform float blend_alpha;
                uniform float mask_factor;
                uniform bool use_tonemap;
                uniform int display_mode;
                uniform vec3 light_direction;
                uniform float light_size;
                in vec2 uv;
                out vec4 f_color;
                vec3 aces (vec3 x) {
                    float a = 2.51;
                    float b = 0.03;
                    float c = 2.43;
                    float d = 0.59;
                    float e = 0.14;
                    return clamp((x * ( a * x + b)) / ( x * (c * x + d)+e), 0.0, 1.0);
                }
                void main() {
                    vec3 bg = pow(texture(tex_input, uv).rgb, vec3(2.2));
                    vec3 albedo = pow(texture(tex_albedo, uv).rgb, vec3(2.2));
                    vec3 norm_raw = texture(tex_normal, uv).rgb;
                    float raw_mask = texture(tex_mask, uv).r;
                    float effective_mask = mix(1.0, raw_mask, mask_factor);

                    vc3 normal = normalize(norm_raw * 2.0 -1.0);

                    vec3 light_dir = normalize(light_direction);
                    vec3 view_dir = normalize(vec3(uv, 0.0) - light_pos);

                    float diff = max(dot(normal, light_dir), 0.0);

                    float light_factor = 1.0;
                    if (light_size > 0.0) {
                        float distance_to_light = length(vec3(uv, 0.0) - light_pos);
                        light_factor = 1.0 / (1.0 + light_size * distance_to_light * distance_to_light);
                    }
                    float atten = light_factor;
                    vec3 lit_layer = albedo * (diff * atten * light_color);
                    if (display_mode == 6) {
                        vec3 linear_result = bg +(lit_layer * effective_mask * blend_alpha);
                        vec3 final_color;
                        if (use_tonemap) {
                            final_color = aces(linear_result);
                            final_color = pow(final_color, vec3(1.0/2.2));
                        } else {
                            final_color = pow(linear_result, vec3(1.0/2.2));
                        }
                        f_color = vec4(final_color, 1.0);
                    } 
                    else if (display_mode == 7) {
                        vec3 linear_result = bg + (lit_layer * blend_alpha);
                        vec3 final_color;
                        if (use_tonemap) {
                            final_color = aces(linear_result);
                            final_color = pow(final_color, vec3(1.0/2.2));
                        } else {
                            final_color = pow(linear_result, vec3(1.0/2.2));
                        }
                        f_color = vec4(final_color, 1.0);
                    }
                    else {
                        if (display_mode == 1) f_color=vec4(pow(bg, vec3(1.0/2.2)), 1.0);
                        else if (display_mode == 2) f_color=vec4(vec3(raw_mask), 1.0);
                        else if (display_mode == 3) f_color=vec4(albedo, 1.0);
                        else if (display_mode == 4) f_color=vec4(norm_raw, 1.0);
                        else if (display_mode == 5) f_color=vec4(lit_layer, 1.0);   
                        else f_color=vec4(0.0, 0.0, 0.0, 1.0);
                    }    
                }
            '''
        )
        self.quad_vao = self.create_quad_vao()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(self.window_size, 4)]
        )
        self.init_env_map_cache()

        self.input_path = input_path
        self.mask_path = mask_path
        self.albedo_path = albedo_path
        self.normal_path = normal_path
        self.output_base_path = output_base_path

        self.display_mode = 6
        self.global_alpha = 1.0
        self.mouse_pos = (0.5, 0.5)
        self.use_tonemap = False
        self.PALETTE = [
            (1.0, 0.95, 0.8),
            (1.0, 0.2, 0.1),
            (0.1, 0.6, 1.0),
            (0.5, 1.0, 0.5),
            (1.0, 1.0, 1.0),
        ]
        self.palette_idx = 0
        self.current_color = np.array(self.PALETTE[0])
        self.mask_enabled = True
        self.mask_val_smooth = 1.0

        self.light_direction = (0.0, 0.0, 1.0)
        self.light_size = 0.5

    def init_env_map_cache(self):
        width, height = self.window_size
        x_coords = np.linspace(0, 2*np.pi, width, endpoint=False)
        y_coords = np.linspace(0, np.pi, height, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(x_coords, y_coords)

        dir_x = np.sin(phi_grid) * np.cos(theta_grid)
        dir_y = np.cos(phi_grid)
        dir_z = np.sin(phi_grid) * np.sin(theta_grid)

        self.sample_dirs = np.stack([dir_x, dir_y, dir_z], axis=-1)

    def create_quad_vao(self):
        if self.ctx is None:
            return None
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0, 1.0,
        ], dtype='f4')
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        vbo = self.ctx.buffer(vertices)
        ibo = self.ctx.buffer(indices)
        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f 2f', 'in_position', 'in_texcoord_0')],
            ibo
        )
        return vao  
    def load_texture_2d(self, path):
        if self.ctx is None:
            return None
        img = Image.open(path).convert('RGBA')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        texture = self.ctx.texture(img.size, 4, img.tobytes())
        texture.build_mipmaps()
        return texture
    
    def process_single_image(self, input_file, mask_file, albedo_file, normal_file, output_file, env_map_file, source_copy_file=None):
         tex_input = tex_mask = tex_albedo = tex_normal = None
         try:
             input_img = Image.open(input_file)
             input_width, input_height = input_img.size
             self.window_size = (input_width, input_height)
         except Exception as e:
             print(f"Failed to open input image {input_file}: {e}")
             self.window_size = (512, 512) #依据输入图片尺寸进行修改
        
         if self.copy_source and source_copy_file:
            try:
                os.makedirs(os.path.dirname(source_copy_file), exist_ok=True)
                shutil.copy2(input_file, source_copy_file)
            except Exception as e:
                print(f"Failed to copy source image to {source_copy_file}: {e}")
         elif self.copy_source:
                print(f"Source copy file path not provided, skipping copy for {input_file}")
         if self.ctx is None:
             self.save_light_environment_map(env_map_file)
             return True
         try:
             self.fbo = self.ctx.framebuffer(
                 color_attachments=[self.ctx.renderbuffer(self.window_size, 4)]
            )
             self.init_env_map_cache()
             self.quad_vao = self.create_quad_vao()

             tex_input = self.load_texture_2d(input_file)  
             tex_mask = self.load_texture_2d(mask_file) 
             tex_albedo = self.load_texture_2d(albedo_file)
             tex_normal = self.load_texture_2d(normal_file) 
             if not all([tex_input, tex_mask, tex_albedo, tex_normal]):
                 print(f"Failed to load one or more textures for {input_file}")
                 self.save_light_environment_map(env_map_file)
                 return True
             for tex in [tex_input, tex_mask, tex_albedo, tex_normal]:
                 tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
             tex_input.use(location=0)
             tex_mask.use(location=1)
             tex_albedo.use(location=2)
             tex_normal.use(location=3) 

             self.prog['tex_input'].value = 0
             self.prog['tex_mask'].value = 1
             self.prog['tex_albedo'].value = 2
             self.prog['tex_normal'].value = 3  

             self.prog[tex_input].value = 0
             self.prog[tex_mask].value = 1
             self.prog[tex_albedo].value = 2
             self.prog[tex_normal].value = 3

             target_col = np.array(self.PALETTE[self.palette_idx])
             self.current_color = self.current_color * 0.92 + target_col * 0.08
             target_mask = 1.0 if self.mask_enabled else 0.0
             self.mask_val_smooth = self.mask_val_smooth * 0.9 + target_mask * 0.1

             self.prog['light_color'].value = tuple(self.current_color)
             self.prog['light_pos'].value = (self.mouse_pos[0], self.mouse_pos[1], 0.15)
             self.prog['light_direction'].value = tuple(self.light_direction)
             self.prog['light_size'].value = self.light_size      
             self.prog['blend_alpha'].value = self.global_alpha
             self.prog['mask_factor'].value = self.mask_val_smooth
             self.prog['use_tonemap'].value = self.use_tonemap
             self.prog['display_mode'].value = self.display_mode

             self.fbo.use()
             self.ctx.enable(moderngl.CULL_FACE)
             self.ctx.enable(moderngl.DEPTH_TEST)
             self.ctx.clear(0.0, 0.0, 0.0)

             self.quad_vao.render(moderngl.TRIANGLES)

             os.makedirs(os.path.dirname(output_file), exist_ok=True)
             self.save_framebuffer_as_image(output_file)

             os.makedirs(os.path.dirname(env_map_file), exist_ok=True)
             self.save_framebuffer_as_image(env_map_file)

             return True
         except Exception as e:
            print(f"Error processing image {input_file}: {e}")
            try:
                os.makedirs(os.path.dirname(env_map_file), exist_ok=True)
                self.save_light_environment_map(env_map_file)
            except:
                pass
            return False
         finally:
             if tex_input:
                 tex_input.release()
             if tex_mask:
                 tex_mask.release()
             if tex_albedo:
                 tex_albedo.release()
             if tex_normal:
                 tex_normal.release()       
    
    def batch_process(self):
        input_files = []
        for file in os.listdir(self.input_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_files.append(file)
        if not input_files:
            print("No input images found.")
            return
        print(f"Found {len(input_files)} input images. Starting processing...")
        success_count = 0
        fail_count = 0

        for filename in tqdm(input_files, desc="Processing images"):
            try:
                input_file = os.path.join(self.input_path, filename)
                base_name = os.path.splitext(filename)[0]
                ext = '.png'

                output_filename = base_name.replace('_Source', '_Render') + ext
                output_file = os.path.join(self.output_base_path, 'Render', output_filename)
                env_map_filename = base_name.replace('_Source', '_HDRI') + ext
                env_map_file = os.path.join(self.output_base_path, 'HDRI', env_map_filename)

                mask_filename = filename.replace('_Source', '_Alpha') 
                albedo_filename = filename.replace('_Source', '_Albedo') 
                normal_filename = filename.replace('_Source', '_Normal')              
                mask_file = os.path.join(self.mask_path, mask_filename)
                albedo_file = os.path.join(self.albedo_path, albedo_filename)
                normal_file = os.path.join(self.normal_path, normal_filename) 
                source_copy_file = None
                if self.copy_source:
                    source_copy_filename = base_name.replace('_Source_', '_Source_') + ext
                    source_copy_file = os.path.join(self.output_base_path, 'Source', source_copy_filename)
                if self.process_single_image(input_file, mask_file, albedo_file, normal_file, output_file, env_map_file, source_copy_file):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Unexpected error processing {filename}: {e}")
                fail_count += 1     
        print(f"Processing completed: {success_count} succeeded, {fail_count} failed.")

    def save_framebuffer_as_image(self, filename):
        if self.ctx is None:
            return
        width, height = self.window_size
        buffer = self.ctx.fbo.read(components=4)
        img = Image.frombuffer('RGBA', (width, height), buffer, 'raw', 'RGBA', 0, 1)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filename)
    
    def save_light_environment_map(self, filename=None):
        try:
            width, height = self.window_size
            sample_dirs = self.sample_dirs

            light_dir = np.array(self.light_direction)
            light_color = np.array(self.current_color)
            light_intensity = self.global_alpha
            light_size = self.light_size

            dot_products = np.einsum('ijk,k->ij', sample_dirs, light_dir)
            base_intensity = np.maximum(dot_products, 0.0)

            if light_size > 0:
                cos_angle = np.einsum('ijk,k->ij', sample_dirs, light_dir)
                angles = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                angle_attenuation = np.exp(-angles*light_size*2.0)

                base_intensity *= angle_attenuation
            final_intensity = base_intensity + light_intensity
            env_colors = final_intensity[..., np.newaxis] * light_color

            env_img_array = np.clip(env_colors*255.0, 0, 255).astype(np.uint8)
            env_img = Image.fromarray(env_img_array, 'RGB')
            if filename:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                env_img.save(filename)
            else:
                default_path = "products/default_env_map.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                env_img.save(default_path)
                print(f"Environment map saved to default path: {default_path}")     
        except Exception as e:
            print(f"Failed to save environment map: {e}")      

def main():
    base_path = "path/to/dataset"
    input_path = os.path.join(base_path, "Source")
    mask_path = os.path.join(base_path, "Alpha")
    albedo_path = os.path.join(base_path, "Albedo")
    normal_path = os.path.join(base_path, "Normal")
    output_base_path = os.path.join(base_path, "output")    

    renderer = FilmicLightRenderer(
        input_path=input_path,
        mask_path=mask_path,
        albedo_path=albedo_path,
        normal_path=normal_path,
        output_base_path=output_base_path,
        copy_source=True
    )

    renderer.batch_process()    

if __name__ == "__main__":
    main()     
