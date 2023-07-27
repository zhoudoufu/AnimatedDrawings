import fire
import yaml
import re
from collections import defaultdict
from pathlib import Path
import os


class BvhNode:

    def __init__(self, value=[], parent=None):
        self.value = value
        self.children = []
        self.parent = parent
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item):
        item.parent = self
        self.children.append(item)

    def filter(self, key):
        for child in self.children:
            if child.value[0] == key:
                yield child

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        return self.value[1]


class Bvh:

    def __init__(self, data):
        self.data = data
        self.root = BvhNode()
        self.frames = []
        self.tokenize()

    def tokenize(self):
        first_round = []
        accumulator = ''
        for char in self.data:
            if char not in ('\n', '\r'):
                accumulator += char
            elif accumulator:
                first_round.append(re.split('\\s+', accumulator.strip()))
                accumulator = ''
        node_stack = [self.root]
        frame_time_found = False
        node = None
        for item in first_round:
            if frame_time_found:
                self.frames.append(item)
                continue
            key = item[0]
            if key == '{':
                node_stack.append(node)
            elif key == '}':
                node_stack.pop()
            else:
                node = BvhNode(item)
                node_stack[-1].add_child(node)
            if item[0] == 'Frame' and item[1] == 'Time:':
                frame_time_found = True

    def search(self, *items):
        found_nodes = []

        def check_children(node):
            if len(node.value) >= len(items):
                failed = False
                for index, item in enumerate(items):
                    if node.value[index] != item:
                        failed = True
                        break
                if not failed:
                    found_nodes.append(node)
            for child in node:
                check_children(child)
        check_children(self.root)
        return found_nodes

    def get_joints(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint)
            for child in joint.filter('JOINT'):
                iterate_joints(child)
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def get_joints_names(self):
        joints = []

        def iterate_joints(joint):
            joints.append(joint.value[1])
            for child in joint.filter('JOINT'):
                iterate_joints(child)
        iterate_joints(next(self.root.filter('ROOT')))
        return joints

    def joint_direct_children(self, name):
        joint = self.get_joint(name)
        return [child for child in joint.filter('JOINT')]

    def get_joint_index(self, name):
        return self.get_joints().index(self.get_joint(name))

    def get_joint(self, name):
        found = self.search('ROOT', name)
        if not found:
            found = self.search('JOINT', name)
        if found:
            return found[0]
        raise LookupError('joint not found')

    def joint_offset(self, name):
        joint = self.get_joint(name)
        offset = joint['OFFSET']
        return (float(offset[0]), float(offset[1]), float(offset[2]))

    def joint_channels(self, name):
        joint = self.get_joint(name)
        return joint['CHANNELS'][1:]

    def get_joint_channels_index(self, joint_name):
        index = 0
        for joint in self.get_joints():
            if joint.value[1] == joint_name:
                return index
            index += int(joint['CHANNELS'][0])
        raise LookupError('joint not found')

    def get_joint_channel_index(self, joint, channel):
        channels = self.joint_channels(joint)
        if channel in channels:
            channel_index = channels.index(channel)
        else:
            channel_index = -1
        return channel_index

    def frame_joint_channel(self, frame_index, joint, channel, value=None):
        joint_index = self.get_joint_channels_index(joint)
        channel_index = self.get_joint_channel_index(joint, channel)
        if channel_index == -1 and value is not None:
            return value
        return float(self.frames[frame_index][joint_index + channel_index])

    def frame_joint_channels(self, frame_index, joint, channels, value=None):
        values = []
        joint_index = self.get_joint_channels_index(joint)
        for channel in channels:
            channel_index = self.get_joint_channel_index(joint, channel)
            if channel_index == -1 and value is not None:
                values.append(value)
            else:
                values.append(
                    float(
                        self.frames[frame_index][joint_index + channel_index]
                    )
                )
        return values

    def frames_joint_channels(self, joint, channels, value=None):
        all_frames = []
        joint_index = self.get_joint_channels_index(joint)
        for frame in self.frames:
            values = []
            for channel in channels:
                channel_index = self.get_joint_channel_index(joint, channel)
                if channel_index == -1 and value is not None:
                    values.append(value)
                else:
                    values.append(
                        float(frame[joint_index + channel_index]))
            all_frames.append(values)
        return all_frames

    def joint_parent(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return None
        return joint.parent

    def joint_parent_index(self, name):
        joint = self.get_joint(name)
        if joint.parent == self.root:
            return -1
        return self.get_joints().index(joint.parent)

    @property
    def nframes(self):
        try:
            return int(next(self.root.filter('Frames:')).value[1])
        except StopIteration:
            raise LookupError('number of frames not found')

    @property
    def frame_time(self):
        try:
            return float(next(self.root.filter('Frame')).value[2])
        except StopIteration:
            raise LookupError('frame time not found')

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            return cls(f.read())


def load_char(yaml_fp):
    with open(yaml_fp, 'r') as f:
        base_cfg = defaultdict(dict, yaml.load(
            f, Loader=yaml.FullLoader) or {})
    return base_cfg


def format_bvh_from_mixamo(bvh_fp):
    # replace text
    with open(bvh_fp, 'r') as f:
        bvh = f.read()
    if 'mixamorig:' in bvh:
        bvh = bvh.replace('mixamorig:', '')
        with open(bvh_fp, 'w') as f:
            f.write(bvh)


def generate_motion_yaml_mixamo(bvh_fp, motion_yaml_dir, base_motion_cfg):
    motion_name = Path(bvh_fp).stem
    motion_yaml = os.path.join(motion_yaml_dir, motion_name + ".yaml")
    if not os.path.exists(motion_yaml):
        # generate motion yaml
        with open(base_motion_cfg, 'r') as f:
            motion_cfg = yaml.load(
                f, Loader=yaml.FullLoader)
        motion_cfg['filepath'] = bvh_fp
        with open(motion_yaml, 'w') as f:
            yaml.dump(motion_cfg, f)
    else:
        print("motion yaml exists: ", motion_yaml)

    return motion_yaml


def generate_mvc_yaml_mixamo(motion_yaml, char_yaml, retarget_yaml, mvc_dir):
    mvc_path = os.path.join(mvc_dir, Path(motion_yaml).stem + ".yaml")
    json_obj = {
        "scene": {
            "ANIMATED_CHARACTERS": [{
                "character_cfg": char_yaml,
                "motion_cfg": motion_yaml,
                "retarget_cfg": retarget_yaml
            }]
        }
    }
    with open(mvc_path, 'w') as f:
        yaml.dump(json_obj, f)
    return mvc_path


def generate_from_mixamo_bvh(bvh_fp="examples/bvh/mixamo/breakdanceFreezeVar.bvh",
                             char_yaml="examples/characters/char3/char_cfg.yaml"
                             ):
    motion_yaml_dir = "examples/config/motion/"
    base_motion_cfg = "examples/config/motion/hiphop.yaml"
    mvc_dir = "examples/config/mvc/"

    format_bvh_from_mixamo(bvh_fp)
    motion_yaml = generate_motion_yaml_mixamo(
        bvh_fp, motion_yaml_dir, base_motion_cfg)

    retarget_yaml = "examples/config/retarget/mixamo_manual.yaml"
    mvc_path = generate_mvc_yaml_mixamo(
        motion_yaml, char_yaml, retarget_yaml, mvc_dir)
    print("mvc_path: ", mvc_path)
    return mvc_path


"""
m1 = Bvh.from_file('examples/bvh/rokoko/jesse_dance.bvh')
m2 = Bvh.from_file('examples/bvh/mixamo/hiphop.bvh')
char_cfg = load_char('examples/characters/char3/char_cfg.yaml')
char_joints = [i["name"] for i in char_cfg['skeleton'] if i["name"] != 'root']
print('done')
"""
if __name__ == '__main__':
    fire.Fire(
        {
            "generate_from_mixamo_bvh": generate_from_mixamo_bvh
        }
    )
