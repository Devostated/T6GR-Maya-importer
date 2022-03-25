import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMaya as OpenMaya
import maya.OpenMayaAnim as OpenMayaAnim
import maya.OpenMayaMPx as OpenMayaMPx
from urllib.request import urlopen
import sys
import os
import os.path
import math
import time
import struct
import json
from copy import copy


#  ____  _____ __  __           _      _ 
# / ___|| ____|  \/  | ___   __| | ___| |
# \___ \|  _| | |\/| |/ _ \ / _` |/ _ \ |
#  ___) | |___| |  | | (_) | (_| |  __/ |
# |____/|_____|_|  |_|\___/ \__,_|\___|_|
                                        
LOG_READ_TIME = False
LOG_WRITE_TIME = False

try:
    xrange
except NameError:
    xrange = range

def enum(**enums):
    return type('Enum', (), enums)

SEMODEL_PRESENCE_FLAGS = enum(
    # Whether or not this model contains a bone block
    SEMODEL_PRESENCE_BONE=1 << 0,
    # Whether or not this model contains submesh blocks
    SEMODEL_PRESENCE_MESH=1 << 1,
    # Whether or not this model contains inline material blocks
    SEMODEL_PRESENCE_MATERIALS=1 << 2,

    # The file contains a custom data block
    SEMODEL_PRESENCE_CUSTOM=1 << 7,
)

SEMODEL_BONEPRESENCE_FLAGS = enum(
    # Whether or not bones contain global-space matrices
    SEMODEL_PRESENCE_GLOBAL_MATRIX=1 << 0,
    # Whether or not bones contain local-space matrices
    SEMODEL_PRESENCE_LOCAL_MATRIX=1 << 1,

    # Whether or not bones contain scales
    SEMODEL_PRESENCE_SCALES=1 << 2,
)

SEMODEL_MESHPRESENCE_FLAGS = enum(
    # Whether or not meshes contain at least 1 uv map
    SEMODEL_PRESENCE_UVSET=1 << 0,
    # Whether or not meshes contain vertex normals
    SEMODEL_PRESENCE_NORMALS=1 << 1,
    # Whether or not meshes contain vertex colors (RGBA)
    SEMODEL_PRESENCE_COLOR=1 << 2,
    # Whether or not meshes contain at least 1 weighted skin
    SEMODEL_PRESENCE_WEIGHTS=1 << 3,
)


class Info(object):
    __slots__ = ('version', 'magic')

    def __init__(self, file=None):
        self.version = 1
        self.magic = b'SEModel'
        if file is not None:
            self.load(file)

    def load(self, file):
        bytes = file.read(9)
        data = struct.unpack('=7ch', bytes)

        magic = b''
        for i in range(7):
            magic += data[i]

        version = data[7]

        assert magic == self.magic
        assert version == self.version

    def save(self, file):
        bytes = self.magic
        bytes += struct.pack('h', self.version)
        file.write(bytes)


class Header(object):
    __slots__ = (
        'dataPresenceFlags', 'bonePresenceFlags',
        'meshPresenceFlags', 'boneCount',
        'meshCount', 'matCount',
    )

    def __init__(self, file=None):
        self.dataPresenceFlags = 0x0
        self.bonePresenceFlags = 0x0
        self.meshPresenceFlags = 0x0

        self.boneCount = 0
        self.meshCount = 0
        self.matCount = 0

        if file is not None:
            self.load(file)

    def load(self, file):
        bytes = file.read(2)
        data = struct.unpack('h', bytes)

        headerSize = data[0]
        bytes = file.read(headerSize - 2)
        # = prefix tell is to ignore C struct packing rules
        data = struct.unpack('=3BIII3B', bytes)

        self.dataPresenceFlags = data[0]
        self.bonePresenceFlags = data[1]
        self.meshPresenceFlags = data[2]

        self.boneCount = data[3]
        self.meshCount = data[4]
        self.matCount = data[5]
        # reserved = data[6]
        # reserved = data[7]
        # reserved = data[8]

    def save(self, file):
        bytes = struct.pack('=3BIII3B',
                            self.dataPresenceFlags, self.bonePresenceFlags,
                            self.meshPresenceFlags, self.boneCount,
                            self.meshCount, self.matCount,
                            0, 0, 0)

        size = struct.pack('h', len(bytes) + 2)
        file.write(size)
        file.write(bytes)


class Bone_t(object):
    """
    The Bone_t class is only ever used to get the size
    and format character used by weight indices in the semodel
    """
    __slots__ = ('size', 'char')

    def __init__(self, header):
        if header.boneCount <= 0xFF:
            self.size = 1
            self.char = 'B'
        elif header.boneCount <= 0xFFFF:
            self.size = 2
            self.char = 'H'
        else:  # if header.boneCount <= 0xFFFFFFFF:
            self.size = 4
            self.char = 'I'


class Face_t(object):
    """
    The Face_t class is only ever used to get the size
    and format character used by face indices in the semodel
    """
    __slots__ = ('size', 'char')

    def __init__(self, mesh):
        if mesh.vertexCount <= 0xFF:
            self.size = 1
            self.char = 'B'
        elif mesh.vertexCount <= 0xFFFF:
            self.size = 2
            self.char = 'H'
        else:
            self.size = 4
            self.char = 'I'


class SimpleMaterialData(object):
    __slots__ = ('diffuseMap', 'normalMap', 'specularMap')

    def __init__(self, file=None):
        self.diffuseMap = ""
        self.normalMap = ""
        self.specularMap = ""

        if file is not None:
            self.load(file)

    def load(self, file):
        # Diffuse map image
        bytes = b''
        b = file.read(1)
        while not b == b'\x00':
            bytes += b
            b = file.read(1)
        self.diffuseMap = bytes.decode("utf-8")
        # Normal map image
        bytes = b''
        b = file.read(1)
        while not b == b'\x00':
            bytes += b
            b = file.read(1)
        self.normalMap = bytes.decode("utf-8")
        # Specular map image
        bytes = b''
        b = file.read(1)
        while not b == b'\x00':
            bytes += b
            b = file.read(1)
        self.specularMap = bytes.decode("utf-8")

    def save(self, file):
        # Diffuse map image
        bytes = struct.pack('%ds' % (len(self.diffuseMap) + 1),
                            self.diffuseMap.encode())
        file.write(bytes)
        # Normal map image
        bytes = struct.pack('%ds' % (len(self.normalMap) + 1),
                            self.normalMap.encode())
        file.write(bytes)
        # Specular map image
        bytes = struct.pack('%ds' % (len(self.specularMap) + 1),
                            self.specularMap.encode())
        file.write(bytes)


class Material(object):
    __slots__ = ('name', 'isSimpleMaterial', 'inputData')

    def __init__(self, file=None):
        self.name = ""
        self.isSimpleMaterial = True
        self.inputData = SimpleMaterialData()

        if file is not None:
            self.load(file)

    def load(self, file):
        bytes = b''
        b = file.read(1)
        while not b == b'\x00':
            bytes += b
            b = file.read(1)
        # Decode name
        self.name = bytes.decode("utf-8")
        # Are we a simple material
        self.isSimpleMaterial = struct.unpack("?", file.read(1))[0]

        # If simple material, decode simple payload
        if (self.isSimpleMaterial):
            self.inputData = SimpleMaterialData(file)

    def save(self, file):
        bytes = struct.pack('%dsB' % (len(self.name) + 1),
                            self.name.encode(), self.isSimpleMaterial)
        file.write(bytes)

        # Ask the input to save itself
        self.inputData.save(file)


class Bone(object):
    __slots__ = ('name', 'flags',
                 'boneParent', 'globalPosition', 'globalRotation',
                 'localPosition', 'localRotation',
                 'scale')

    def __init__(self, file=None):
        self.name = ""

        self.flags = 0x0

        self.globalPosition = (0, 0, 0)
        self.globalRotation = (0, 0, 0, 1)

        self.localPosition = (0, 0, 0)
        self.localRotation = (0, 0, 0, 1)

        self.scale = (1, 1, 1)

        if file is not None:
            self.load(file)

    def load(self, file):
        bytes = b''
        b = file.read(1)
        while not b == b'\x00':
            bytes += b
            b = file.read(1)
        self.name = bytes.decode("utf-8")

    def loadData(self, file,
                 useGlobal=False, useLocal=False, useScale=False):
        # Read the flags and boneParent for the bone
        bytes = file.read(5)
        data = struct.unpack("=Bi", bytes)
        self.flags = data[0]
        self.boneParent = data[1]

        # Read global matrices if available
        if useGlobal:
            bytes = file.read(28)
            data = struct.unpack("=7f", bytes)
            self.globalPosition = (data[0], data[1], data[2])
            self.globalRotation = (data[3], data[4], data[5], data[6])

        # Read local matrices if available
        if useLocal:
            bytes = file.read(28)
            data = struct.unpack("=7f", bytes)
            self.localPosition = (data[0], data[1], data[2])
            self.localRotation = (data[3], data[4], data[5], data[6])

        # Read scale if available
        if useScale:
            bytes = file.read(12)
            data = struct.unpack("=3f", bytes)
            self.scale = (data[0], data[1], data[2])

    def save(self, file,
             useGlobal=False, useLocal=False, useScale=False):
        bytes = struct.pack("=Bi", self.flags, self.boneParent)
        file.write(bytes)

        if useGlobal:
            bytes = struct.pack("=7f", self.globalPosition[0], self.globalPosition[1],
                                self.globalPosition[2], self.globalRotation[0],
                                self.globalRotation[1], self.globalRotation[2],
                                self.globalRotation[3])
            file.write(bytes)

        if useLocal:
            bytes = struct.pack("=7f", self.localPosition[0], self.localPosition[1],
                                self.localPosition[2], self.localRotation[0],
                                self.localRotation[1], self.localRotation[2],
                                self.localRotation[3])
            file.write(bytes)

        if useScale:
            bytes = struct.pack(
                "=3f", self.scale[0], self.scale[1], self.scale[2])
            file.write(bytes)


class Vertex(object):
    __slots__ = ('position', 'uvLayers', 'normal', 'color', 'weights')

    def __init__(self, uvSetCount=0, maxSkinInfluence=0):
        self.position = (0, 0, 0)
        self.normal = (0, 0, 0)
        self.color = (1, 1, 1, 1)

        self.uvLayers = [(0, 0)] * uvSetCount
        self.weights = [(0, 0)] * maxSkinInfluence

    @staticmethod
    def loadData(file, vertexCount, bone_t,
                 uvSetCount=0, maxSkinInfluence=0,
                 useNormals=False, useColors=False):
        # Preallocate verticies
        vertex_buffer = [None] * vertexCount

        # Positions first
        bytes = file.read(12 * vertexCount)
        data_pos = struct.unpack("=%df" % (3 * vertexCount), bytes)

        # UVLayers
        bytes = file.read((8 * uvSetCount) * vertexCount)
        data_uvs = struct.unpack(
            "=%df" % ((2 * uvSetCount) * vertexCount), bytes)

        # Normals
        if useNormals:
            bytes = file.read(12 * vertexCount)
            data_norms = struct.unpack("=%df" % (3 * vertexCount), bytes)

        # Colors
        if useColors:
            bytes = file.read(4 * vertexCount)
            data_colors = struct.unpack("=%dB" % (4 * vertexCount), bytes)

        # Weights
        bytes = file.read(((4 + bone_t.size) * maxSkinInfluence) * vertexCount)
        data_weights = struct.unpack(
            "=" + ((("%cf" % bone_t.char) * maxSkinInfluence) * vertexCount), bytes)

        for vert_idx in xrange(vertexCount):
            # Initialize vertex, assign position
            vertex_buffer[vert_idx] = Vertex(
                uvSetCount=uvSetCount, maxSkinInfluence=maxSkinInfluence)
            vertex_buffer[vert_idx].position = data_pos[vert_idx *
                                                        3:(vert_idx * 3) + 3]

            if uvSetCount > 0:
                uv_layers = data_uvs[vert_idx *
                                     (2 * uvSetCount):(vert_idx *
                                                       (2 * uvSetCount)) + 2 * uvSetCount]
                for uvi in xrange(uvSetCount):
                    vertex_buffer[vert_idx].uvLayers[uvi] = uv_layers[uvi *
                                                                      2:(uvi * 2) + 2]

            if useNormals:
                vertex_buffer[vert_idx].normal = data_norms[vert_idx *
                                                            3:(vert_idx * 3) + 3]
            if useColors:
                color = data_colors[vert_idx * 4:(vert_idx * 4) + 4]
                vertex_buffer[vert_idx].color = (
                    color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255)

            if maxSkinInfluence > 0:
                weights = data_weights[vert_idx *
                                       (2 * maxSkinInfluence):(vert_idx *
                                                               (2 * maxSkinInfluence)) + (2 * maxSkinInfluence)]
                for weight in xrange(maxSkinInfluence):
                    vertex_buffer[vert_idx].weights[weight] = weights[weight *
                                                                      2:(weight * 2) + 2]

        return vertex_buffer

    def savePosition(self, file):
        bytes = struct.pack(
            "=3f", self.position[0], self.position[1], self.position[2])
        file.write(bytes)

    def saveUVLayers(self, file, matReferenceCount):
        for _idx in xrange(matReferenceCount):
            if _idx < len(self.uvLayers):
                bytes = struct.pack(
                    "=2f", self.uvLayers[_idx][0], self.uvLayers[_idx][1])
                file.write(bytes)
            else:
                bytes = struct.pack(
                    "=2f", 0, 0)
                file.write(bytes)

    def saveNormal(self, file):
        bytes = struct.pack(
            "=3f", self.normal[0], self.normal[1], self.normal[2])
        file.write(bytes)

    def saveColor(self, file):
        bytes = struct.pack(
            "=4B", self.color[0] * 255, self.color[1] * 255, self.color[2] * 255, self.color[3] * 255)
        file.write(bytes)

    def saveWeights(self, file, maxSkinInfluence, bone_t):
        for _idx in xrange(maxSkinInfluence):
            if _idx < len(self.weights):
                bytes = struct.pack(
                    "=%cf" % bone_t.char, self.weights[_idx][0], self.weights[_idx][1])
                file.write(bytes)
            else:
                bytes = struct.pack(
                    "=%cf" % bone_t.char, 0, 0)
                file.write(bytes)


class Face(object):
    __slots__ = ('indices')

    def __init__(self, indices=(0, 1, 2)):
        self.indices = indices

    @staticmethod
    def loadData(file, faceCount, face_t):
        # Load variable length face buffer
        bytes = file.read((3 * face_t.size) * faceCount)
        data = struct.unpack("=%d%c" % ((faceCount * 3), face_t.char), bytes)

        # Create and return face buffer
        face_buffer = [None] * faceCount
        for face_idx, face_data in enumerate((data[i:i + 3] for i in xrange(0, len(data), 3))):
            face_buffer[face_idx] = Face(face_data)

        return face_buffer

    def save(self, file, face_t):
        bytes = struct.pack("=3%c" % face_t.char,
                            self.indices[0], self.indices[1], self.indices[2])
        file.write(bytes)


class Mesh(object):
    __slots__ = ('flags', 'vertexCount', 'faceCount',
                 'vertices', 'faces',
                 'materialReferences', 'matReferenceCount',
                 'maxSkinInfluence')

    def __init__(self, file=None, bone_t=None,
                 useUVs=False, useNormals=False,
                 useColors=False, useWeights=False):
        self.flags = 0x0

        self.vertexCount = 0
        self.faceCount = 0

        self.vertices = []
        self.faces = []

        self.matReferenceCount = 0
        self.maxSkinInfluence = 0

        self.materialReferences = []

        if file is not None:
            self.load(file, bone_t, useUVs, useNormals, useColors, useWeights)

    def load(self, file, bone_t,
             useUVs=False, useNormals=False,
             useColors=False, useWeights=False):
        bytes = file.read(11)
        data = struct.unpack("=3BII", bytes)
        self.flags = data[0]

        self.matReferenceCount = data[1]
        self.maxSkinInfluence = data[2]

        if not useUVs:
            self.matReferenceCount = 0
        if not useWeights:
            self.maxSkinInfluence = 0

        self.vertexCount = data[3]
        self.faceCount = data[4]

        # Expand buffers before loading
        self.materialReferences = [None] * self.matReferenceCount

        # Calculate face index
        face_t = Face_t(self)

        # Load vertex buffer
        self.vertices = Vertex.loadData(file, self.vertexCount, bone_t,
                                        self.matReferenceCount, self.maxSkinInfluence,
                                        useNormals, useColors)

        # Load face buffer
        self.faces = Face.loadData(file, self.faceCount, face_t)

        # Load material reference buffer (signed int32_t's per count)
        for mat_idx in xrange(self.matReferenceCount):
            self.materialReferences[mat_idx] = struct.unpack("i", file.read(4))[
                0]

    def save(self, file, bone_t, useUVs=False, useNormals=False, useColors=False, useWeights=False):
        # Update metadata first
        self.vertexCount = len(self.vertices)
        self.faceCount = len(self.faces)

        face_t = Face_t(self)

        for vertex in self.vertices:
            if len(vertex.uvLayers) > self.matReferenceCount:
                self.matReferenceCount = len(vertex.uvLayers)
            if len(vertex.weights) > self.maxSkinInfluence:
                self.maxSkinInfluence = len(vertex.weights)

        # Ensure we have enough references per layer, if not, default to no material
        if len(self.materialReferences) < self.matReferenceCount:
            for _idx in xrange(self.matReferenceCount - len(self.materialReferences)):
                self.materialReferences.append(-1)

        bytes = struct.pack("=3BII", self.flags, self.matReferenceCount,
                            self.maxSkinInfluence, self.vertexCount, self.faceCount)
        file.write(bytes)

        # Produce vertex buffer by data type
        for vertex in self.vertices:
            vertex.savePosition(file)

        if useUVs:
            for vertex in self.vertices:
                vertex.saveUVLayers(file, self.matReferenceCount)

        if useNormals:
            for vertex in self.vertices:
                vertex.saveNormal(file)

        if useColors:
            for vertex in self.vertices:
                vertex.saveColor(file)

        if useWeights:
            for vertex in self.vertices:
                vertex.saveWeights(file, self.maxSkinInfluence, bone_t)

        # Produce the face buffer
        for face in self.faces:
            face.save(file, face_t)

        # Produce material indices
        for matIndex in self.materialReferences:
            bytes = struct.pack("i", matIndex)
            file.write(bytes)


class SEModel(object):
    __slots__ = ('__info', 'info', 'header', 'bones', 'meshes', 'materials')

    def __init__(self, path=None):
        self.__info = Info()
        self.header = Header()

        self.bones = []
        self.meshes = []
        self.materials = []

        if path is not None:
            self.load(path)

    def update_metadata(self):
        header = self.header
        header.boneCount = len(self.bones)
        header.meshCount = len(self.meshes)
        header.matCount = len(self.materials)

        dataPresenceFlags = header.dataPresenceFlags
        bonePresenceFlags = header.bonePresenceFlags
        meshPresenceFlags = header.meshPresenceFlags

        if header.boneCount:
            dataPresenceFlags |= SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_BONE
        if header.meshCount:
            dataPresenceFlags |= SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_MESH
        if header.matCount:
            dataPresenceFlags |= SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_MATERIALS

        # Check for non-default scale, local, global values
        useScales = False
        useLocals = False
        useGlobals = False

        for bone in self.bones:
            if bone.scale != (1, 1, 1):
                useScales = True
            if bone.localPosition != (0, 0, 0) or bone.localRotation != (0, 0, 0, 1):
                useLocals = True
            if bone.globalPosition != (0, 0, 0) or bone.globalRotation != (0, 0, 0, 1):
                useGlobals = True

            # Check to end early
            if useScales and useLocals and useGlobals:
                break

        if useScales:
            bonePresenceFlags |= SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_SCALES
        if useLocals:
            bonePresenceFlags |= SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_LOCAL_MATRIX
        if useGlobals:
            bonePresenceFlags |= SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_GLOBAL_MATRIX

        # Check for non-default properties
        useNormals = False
        useColors = False
        useUVSet = False
        useWeights = False

        for mesh in self.meshes:
            for vertex in mesh.vertices:
                if len(vertex.uvLayers):
                    useUVSet = True
                if len(vertex.weights):
                    useWeights = True
                if vertex.color != (1, 1, 1, 1):
                    useColors = True
                if vertex.normal != (0, 0, 0):
                    useNormals = True

                # Check to end early
                if useNormals and useColors and useUVSet and useWeights:
                    break

            # Check to end early
            if useNormals and useColors and useUVSet and useWeights:
                break

        if useNormals:
            meshPresenceFlags |= SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_NORMALS
        if useColors:
            meshPresenceFlags |= SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_COLOR
        if useUVSet:
            meshPresenceFlags |= SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_UVSET
        if useWeights:
            meshPresenceFlags |= SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_WEIGHTS

        # Assign header values
        header.dataPresenceFlags = dataPresenceFlags
        header.bonePresenceFlags = bonePresenceFlags
        header.meshPresenceFlags = meshPresenceFlags

    def load(self, path):
        if LOG_READ_TIME:
            time_start = time.time()
            print("Loading: '%s'" % path)

        try:
            file = open(path, "rb")
        except IOError:
            print("Could not open file for reading:\n %s" % path)
            return

        self.info = Info(file)
        self.header = Header(file)

        # Init the bone_t info
        bone_t = Bone_t(self.header)

        dataPresenceFlags = self.header.dataPresenceFlags
        bonePresenceFlags = self.header.bonePresenceFlags
        meshPresenceFlags = self.header.meshPresenceFlags

        self.bones = [None] * self.header.boneCount
        if dataPresenceFlags & SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_BONE:
            useGlobal = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_GLOBAL_MATRIX
            useLocal = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_LOCAL_MATRIX
            useScale = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_SCALES

            # Load bone tag names
            for i in xrange(self.header.boneCount):
                self.bones[i] = Bone(file)

            # Load bone data
            for i in xrange(self.header.boneCount):
                self.bones[i].loadData(
                    file, useGlobal, useLocal, useScale)

        self.meshes = [None] * self.header.meshCount
        if dataPresenceFlags & SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_MESH:
            useUVs = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_UVSET
            useNormals = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_NORMALS
            useColors = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_COLOR
            useWeights = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_WEIGHTS

            # Load submeshes
            for i in xrange(self.header.meshCount):
                self.meshes[i] = Mesh(
                    file, bone_t, useUVs, useNormals, useColors, useWeights)

        # Load materials
        self.materials = [None] * self.header.matCount
        if dataPresenceFlags & SEMODEL_PRESENCE_FLAGS.SEMODEL_PRESENCE_MATERIALS:
            # Load material entries
            for i in xrange(self.header.matCount):
                self.materials[i] = Material(file)

        file.close()

        if LOG_READ_TIME:
            time_end = time.time()
            time_elapsed = time_end - time_start
            print("Done! - Completed in %ss" % time_elapsed)

    def save(self, filepath=""):
        if LOG_WRITE_TIME:
            time_start = time.time()
            print("Saving: '%s'" % filepath)

        try:
            file = open(filepath, "wb")
        except IOError:
            print("Could not open the file for writing:\n %s" % filepath)
            return

        # Update the header flags, based on the presence of different data types (Bones, Meshes, Materials)
        self.update_metadata()

        self.__info.save(file)
        self.header.save(file)

        for bone in self.bones:
            bytes = struct.pack(
                '%ds' % (len(bone.name) + 1), bone.name.encode())
            file.write(bytes)

        bonePresenceFlags = self.header.bonePresenceFlags
        meshPresenceFlags = self.header.meshPresenceFlags

        useGlobals = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_GLOBAL_MATRIX
        useLocals = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_LOCAL_MATRIX
        useScales = bonePresenceFlags & SEMODEL_BONEPRESENCE_FLAGS.SEMODEL_PRESENCE_SCALES

        useUVSet = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_UVSET
        useNormal = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_NORMALS
        useColor = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_COLOR
        useWeights = meshPresenceFlags & SEMODEL_MESHPRESENCE_FLAGS.SEMODEL_PRESENCE_WEIGHTS

        bone_t = Bone_t(self.header)

        for bone in self.bones:
            bone.save(file, useGlobals, useLocals, useScales)

        for mesh in self.meshes:
            mesh.save(file, bone_t, useUVSet, useNormal, useColor, useWeights)

        for mat in self.materials:
            mat.save(file)

        file.close()

        if LOG_WRITE_TIME:
            time_end = time.time()
            time_elapsed = time_end - time_start
            print("Done! - Completed in %ss" % time_elapsed)

#  ____  _____ _____           _       ____  _             _       
# / ___|| ____|_   _|__   ___ | |___  |  _ \| |_   _  __ _(_)_ __  
# \___ \|  _|   | |/ _ \ / _ \| / __| | |_) | | | | |/ _` | | '_ \ 
#  ___) | |___  | | (_) | (_) | \__ \ |  __/| | |_| | (_| | | | | |
# |____/|_____| |_|\___/ \___/|_|___/ |_|   |_|\__,_|\__, |_|_| |_|
#                                                    |___/         

def __scene_bindmesh__(mesh_skin, weight_data):
    """Applies smooth skin bindings"""
    joint_indicies = {}

    # Build a list of API specific joint ids
    _tmp_array = OpenMaya.MDagPathArray()
    mesh_skin.influenceObjects(_tmp_array)
    _tmp_array_len = _tmp_array.length()

    # Iterate and assign indicies for names
    for idx in xrange(_tmp_array_len):
        joint_indicies[str(_tmp_array[idx].fullPathName())
                       ] = mesh_skin.indexForInfluenceObject(_tmp_array[idx])

    # Base format payload
    cluster_attr = str(mesh_skin.name()) + ".weightList[%d]"
    cluster_array_attr = (".weights[0:%d]" % (_tmp_array_len - 1))

    # Buffer for new indicies
    weight_values = [0] * _tmp_array_len

    # Iterate and apply weights
    for vertex, weights in weight_data:
        # Build final string
        if _tmp_array_len == 1:
            weight_payload = cluster_attr % vertex + ".weights[0]"
        else:
            weight_payload = cluster_attr % vertex + cluster_array_attr

        # Iterate over weights per bone
        for joint, weight_val in weights:
            weight_values[joint_indicies[joint]] = weight_val

        # Set all weights at once
        cmds.setAttr(weight_payload, *weight_values)
        # Clear for the next one
        weight_values = [0] * _tmp_array_len


def __scene_newskin__(mesh_path, joints=[], max_influence=1):
    """Creates a skin cluster for the mesh"""
    skin_params = [x for x in joints]
    skin_params.append(mesh_path)

    # Create the skin cluster, maintaining the influence
    try:
        new_skin = cmds.skinCluster(
            *skin_params, tsb=True, mi=max_influence, nw=False)
    except RuntimeError:
        __log_info__(
            "SEModel::NewSkin(%s) could not create a new skinCluster, skipping..." % mesh_path)
        return None

    # Attach the controller, then return
    select_list = OpenMaya.MSelectionList()
    select_list.add(new_skin[0])

    # Attempt to get the path to the first item in the list
    result_cluster = OpenMaya.MObject()
    select_list.getDependNode(0, result_cluster)

    # Return it
    return OpenMayaAnim.MFnSkinCluster(result_cluster)


def __log_info__(format_str=""):
    """Logs a line to the console"""
    print("[SETools] " + format_str)


def __build_image_path__(asset_path, image_path):
    """Builds the full image path"""
    root_path = os.path.dirname(asset_path)
    return os.path.join(root_path, image_path)


def __math_matrixtoquat__(maya_matrix):
    """Converts a Maya matrix array to a quaternion"""
    quat_x, quat_y, quat_z, quat_w = (0, 0, 0, 1)

    trans_remain = maya_matrix[0] + maya_matrix[5] + maya_matrix[10]
    if trans_remain > 0:
        divisor = math.sqrt(trans_remain + 1.0) * 2.0 
        quat_w = 0.25 * divisor
        quat_x = (maya_matrix[6] - maya_matrix[9]) / divisor
        quat_y = (maya_matrix[8] - maya_matrix[2]) / divisor
        quat_z = (maya_matrix[1] - maya_matrix[4]) / divisor
    elif (maya_matrix[0] > maya_matrix[5]) and (maya_matrix[0] > maya_matrix[10]):
        divisor = math.sqrt(
            1.0 + maya_matrix[0] - maya_matrix[5] - maya_matrix[10]) * 2.0
        quat_w = (maya_matrix[6] - maya_matrix[9]) / divisor
        quat_x = 0.25 * divisor
        quat_y = (maya_matrix[4] + maya_matrix[1]) / divisor
        quat_z = (maya_matrix[8] + maya_matrix[2]) / divisor
    elif maya_matrix[5] > maya_matrix[10]:
        divisor = math.sqrt(
            1.0 + maya_matrix[5] - maya_matrix[0] - maya_matrix[10]) * 2.0
        quat_w = (maya_matrix[8] - maya_matrix[2]) / divisor
        quat_x = (maya_matrix[4] + maya_matrix[1]) / divisor
        quat_y = 0.25 * divisor
        quat_z = (maya_matrix[9] + maya_matrix[6]) / divisor
    else:
        divisor = math.sqrt(
            1.0 + maya_matrix[10] - maya_matrix[0] - maya_matrix[5]) * 2.0
        quat_w = (maya_matrix[1] - maya_matrix[4]) / divisor
        quat_x = (maya_matrix[8] + maya_matrix[2]) / divisor
        quat_y = (maya_matrix[9] + maya_matrix[6]) / divisor
        quat_z = 0.25 * divisor

    # Return the result
    return (quat_x, quat_y, quat_z, quat_w)



def __load_semodel__(file_path=""):
    """Imports a SEModel file to the scene"""
    model = SEModel(file_path)

    # We need to configure the scene, save current state and change back later
    autokeyframe_state = cmds.autoKeyframe(query=True)
    currentunit_state = cmds.currentUnit(query=True, linear=True)
    currentangle_state = cmds.currentUnit(query=True, angle=True)
    cmds.autoKeyframe(state=False)
    cmds.currentUnit(linear="cm", angle="deg")

    # Prepare the main progress bar (Requires mel, talk about pathetic)
    #main_progressbar = mel.eval("$tmp = $gMainProgressBar")
    #cmds.progressBar(main_progressbar, edit=True,
    #                 beginProgress=True, isInterruptable=False,
    #                 status='Loading SEModel...', maxValue=max(1,
    #                                                           model.header.boneCount + model.header.meshCount + model.header.matCount))

    # A list of IKJoint handles
    maya_joint_handles = [None] * model.header.boneCount
    maya_joint_paths = [None] * model.header.boneCount
    maya_joints = OpenMaya.MFnTransform()
    # Create root joints node
    maya_joint_node = maya_joints.create()
    maya_joints.setName("Joints")

    # Iterate over the bones and create those first
    for bone_idx, bone in enumerate(model.bones):
        #cmds.progressBar(main_progressbar, edit=True, step=1)

        new_bone = OpenMayaAnim.MFnIkJoint()
        bone_scale = OpenMaya.MScriptUtil()

        # Assign parent via index
        if bone.boneParent <= -1:
            bone_path = new_bone.create(maya_joint_node)
        else:
            bone_path = new_bone.create(maya_joint_handles[bone.boneParent])

        # Rename the joint
        new_bone.setName(bone.name)
        maya_joint_paths[bone_idx] = new_bone.fullPathName()

        # Apply information, note: Check whether or not we have locals/globals
        new_bone.setTranslation(OpenMaya.MVector(
            bone.localPosition[0], bone.localPosition[1], bone.localPosition[2]),
            OpenMaya.MSpace.kTransform)
        new_bone.setOrientation(OpenMaya.MQuaternion(
            bone.localRotation[0], bone.localRotation[1], bone.localRotation[2],
            bone.localRotation[3]))

        # Create and apply scale
        bone_scale.createFromList(
            [bone.scale[0], bone.scale[1], bone.scale[2]], 3)
        new_bone.setScale(bone_scale.asDoublePtr())

        # Store the joint for use later
        maya_joint_handles[bone_idx] = bone_path

    # Iterate over materials and create them next (If they aren't already loaded)
    for mat in model.materials:
        #cmds.progressBar(main_progressbar, edit=True, step=1)

        # Only make if it doesn't exist
        if cmds.objExists(mat.name):
            continue

        # Create the material
        material = cmds.shadingNode("lambert", asShader=True, name=mat.name)
        material_group = cmds.sets(
            renderable=True, empty=True, name=("%sSG" % material))
        # Connect diffuse
        cmds.connectAttr(("%s.outColor" % material),
                         ("%s.surfaceShader" % material_group), force=True)
        # Create diffuse texture
        diffuse_image = cmds.shadingNode(
            'file', name=mat.name + "_c", asTexture=True)
        cmds.setAttr(("%s.fileTextureName" % diffuse_image),
                     __build_image_path__(file_path, mat.inputData.diffuseMap), type="string")
        cmds.connectAttr(("%s.outColor" % diffuse_image),
                         ("%s.color" % material))
        # Connect output mapping
        texture_2d = cmds.shadingNode("place2dTexture", name=(
            "place2dTexture_%s" % (mat.name + "_c")), asUtility=True)
        cmds.connectAttr(("%s.outUV" % texture_2d),
                         ("%s.uvCoord" % diffuse_image))

    # Create the root mesh node
    maya_meshs = OpenMaya.MFnTransform()
    maya_mesh_node = maya_meshs.create()
    maya_meshs.setName(os.path.splitext(os.path.basename(file_path))[0])

    # Iterate over the meshes and create them
    for mesh in model.meshes:
        #cmds.progressBar(main_progressbar, edit=True, step=1)
        # Create root transform
        new_mesh_root = OpenMaya.MFnTransform()
        new_mesh_node = new_mesh_root.create(maya_mesh_node)
        new_mesh_root.setName("SEModelMesh")

        # Perform face validation, maya doesn't like faces with the same verts
        purge_map = []
        for face_idx in xrange(mesh.faceCount):
            face = mesh.faces[face_idx]
            # Compare indicies
            if face.indices[0] == face.indices[1]:
                purge_map.append(face_idx)
            elif face.indices[0] == face.indices[2]:
                purge_map.append(face_idx)
            elif face.indices[1] == face.indices[2]:
                purge_map.append(face_idx)

        # Remove all invalid faces, reverse order to prevent reordering
        for face_idx in reversed(purge_map):
            del mesh.faces[face_idx]
        mesh.faceCount = mesh.faceCount - len(purge_map)

        # Create mesh
        new_mesh = OpenMaya.MFnMesh()
        mesh_vertex_buffer = OpenMaya.MFloatPointArray(mesh.vertexCount)
        mesh_normal_buffer = OpenMaya.MVectorArray(mesh.vertexCount)
        mesh_color_buffer = OpenMaya.MColorArray(
            mesh.vertexCount, OpenMaya.MColor(1, 1, 1, 1))
        mesh_vertex_index = OpenMaya.MIntArray(mesh.vertexCount, 0)
        mesh_face_buffer = OpenMaya.MIntArray(mesh.faceCount * 3)
        mesh_face_counts = OpenMaya.MIntArray(mesh.faceCount, 3)
        mesh_weight_data = [None] * mesh.vertexCount
        mesh_weight_bones = set()

        # Support all possible UV layers
        mesh_uvid_layers = OpenMaya.MIntArray(mesh.faceCount * 3)
        mesh_uvu_layers = []
        mesh_uvv_layers = []

        # We must generate them this way to python doesn't clone an object
        for uv_layer in xrange(mesh.matReferenceCount):
            mesh_uvu_layers.append(OpenMaya.MFloatArray(mesh.faceCount * 3))
            mesh_uvv_layers.append(OpenMaya.MFloatArray(mesh.faceCount * 3))

        # Build buffers
        for vert_idx, vert in enumerate(mesh.vertices):
            mesh_vertex_buffer.set(vert_idx,
                                   vert.position[0], vert.position[1], vert.position[2])
            mesh_normal_buffer.set(OpenMaya.MVector(
                vert.normal[0], vert.normal[1], vert.normal[2]), vert_idx)
            mesh_color_buffer.set(vert_idx,
                                  vert.color[0], vert.color[1], vert.color[2], vert.color[3])
            mesh_vertex_index.set(vert_idx, vert_idx)

            # Build a payload set for this vertex
            vertex_weights = []
            # Iterate and create the weights
            for weight in vert.weights:
                # Weights are valid when value is > 0.0
                if weight[1] > 0.0:
                    mesh_weight_bones.add(maya_joint_paths[weight[0]])
                    vertex_weights.append(
                        (maya_joint_paths[weight[0]], weight[1]))

            # Add the weight set
            mesh_weight_data[vert_idx] = (vert_idx, vertex_weights)

        # Face buffer for maya is inverted 1->0->2
        for face_idx, face in enumerate(mesh.faces):
            mesh_face_buffer.set(face.indices[1], (face_idx * 3))
            mesh_face_buffer.set(face.indices[0], (face_idx * 3) + 1)
            mesh_face_buffer.set(face.indices[2], (face_idx * 3) + 2)

            mesh_uvid_layers.set((face_idx * 3), (face_idx * 3))
            mesh_uvid_layers.set((face_idx * 3) + 1, (face_idx * 3) + 1)
            mesh_uvid_layers.set((face_idx * 3) + 2, (face_idx * 3) + 2)

            # Do this per layer
            for uv_layer in xrange(mesh.matReferenceCount):
                mesh_uvu_layers[uv_layer].set(
                    mesh.vertices[face.indices[1]].uvLayers[uv_layer][0], (face_idx * 3))
                mesh_uvu_layers[uv_layer].set(
                    mesh.vertices[face.indices[0]].uvLayers[uv_layer][0], (face_idx * 3) + 1)
                mesh_uvu_layers[uv_layer].set(
                    mesh.vertices[face.indices[2]].uvLayers[uv_layer][0], (face_idx * 3) + 2)
                mesh_uvv_layers[uv_layer].set(
                    1 - mesh.vertices[face.indices[1]].uvLayers[uv_layer][1], (face_idx * 3))
                mesh_uvv_layers[uv_layer].set(
                    1 - mesh.vertices[face.indices[0]].uvLayers[uv_layer][1], (face_idx * 3) + 1)
                mesh_uvv_layers[uv_layer].set(
                    1 - mesh.vertices[face.indices[2]].uvLayers[uv_layer][1], (face_idx * 3) + 2)

        # Create
        new_mesh.create(mesh.vertexCount, mesh.faceCount, mesh_vertex_buffer, OpenMaya.MIntArray(
            mesh.faceCount, 3), mesh_face_buffer, new_mesh_node)
        # Set normals + colors
        new_mesh.setVertexNormals(mesh_normal_buffer, mesh_vertex_index)
        new_mesh.setVertexColors(mesh_color_buffer, mesh_vertex_index)

        # Apply UVLayers
        for uv_layer in xrange(mesh.matReferenceCount):
            # Use default layer, or, make a new one if need be, following maya names
            if uv_layer > 0:
                new_uv = new_mesh.createUVSetWithName(("map%d" % (uv_layer + 1)))
            else:
                new_uv = new_mesh.currentUVSetName()

            # Set uvs
            new_mesh.setCurrentUVSetName(new_uv)
            new_mesh.setUVs(
                mesh_uvu_layers[uv_layer], mesh_uvv_layers[uv_layer], new_uv)
            new_mesh.assignUVs(
                mesh_face_counts, mesh_uvid_layers, new_uv)

            # Set material, default to shader if not defined
            material_index = mesh.materialReferences[uv_layer]
            try:
                if material_index < 0:
                    cmds.sets(new_mesh.fullPathName(),
                            forceElement="initialShadingGroup")
                else:
                    cmds.sets(new_mesh.fullPathName(), forceElement=(
                        "%sSG" % model.materials[material_index].name))
            except RuntimeError:
                # Occurs when a material was already assigned to the mesh...
                pass


        # Prepare skin weights
        mesh_skin = __scene_newskin__(new_mesh.fullPathName(), [
            x for x in mesh_weight_bones], mesh.maxSkinInfluence)
        if mesh_skin is not None:
            __scene_bindmesh__(mesh_skin, mesh_weight_data)

    # Close the progress bar
    #cmds.progressBar(main_progressbar, edit=True, endProgress=True)

    # Reconfigure the scene to our liking
    cmds.autoKeyframe(state=autokeyframe_state)
    cmds.currentUnit(linear=currentunit_state, angle=currentangle_state)

    # Finished model import
    __log_info__("SEModel::Import(%s) has been imported successfully"
                 % os.path.basename(file_path))



#  _____ __    ____ ____    ____  _                          ____                _           
# |_   _/ /_  / ___|  _ \  | __ )(_)_ __   __ _ _ __ _   _  |  _ \ ___  __ _  __| | ___ _ __ 
#   | || '_ \| |  _| |_) | |  _ \| | '_ \ / _` | '__| | | | | |_) / _ \/ _` |/ _` |/ _ \ '__|
#   | || (_) | |_| |  _ <  | |_) | | | | | (_| | |  | |_| | |  _ <  __/ (_| | (_| |  __/ |   
#   |_| \___/ \____|_| \_\ |____/|_|_| |_|\__,_|_|   \__, | |_| \_\___|\__,_|\__,_|\___|_|   
#                                                    |___/                                   


def read_short(file):
    return struct.unpack("h", file.read(2))[0]

def read_ushort(file):
    return struct.unpack("H", file.read(2))[0]

def read_string(file):
    string = b''
    byte = file.read(1)
    while byte != b'\x00':
        string += byte
        byte = file.read(1)
    return string.decode("latin-1")

def read_byte(file):
    return struct.unpack("B", file.read(1))[0]

def read_int(file):
    return struct.unpack("i", file.read(4))[0]

def read_uint(file):
    return struct.unpack("I", file.read(4))[0]

def read_float(file):
    return struct.unpack("f", file.read(4))[0]



 # _____ __    ____ ____                _           
 #|_   _/ /_  / ___|  _ \ ___  __ _  __| | ___ _ __ 
 #  | || '_ \| |  _| |_) / _ \/ _` |/ _` |/ _ \ '__|
 #  | || (_) | |_| |  _ <  __/ (_| | (_| |  __/ |   
 #  |_| \___/ \____|_| \_\___|\__,_|\__,_|\___|_|   
                                                   

class T6GReader(object):
    """Importer for the .T6GR file format"""

    Playerflag = 0
    CorpseFlag = 1
    KillstreakFlag = 2
    ViewmodelFlag = 3
    ZombieFlag = 4
    ProjectileFlag = 5
    


    FileVersion = 0  
    ModelCount = 0 
    UniqueBoneCount = 0 
    FrameCount = 0 
    FPS = 0 

    ModelDictionary = dict()
    BoneDictionary = dict()
    Frames = list()

    def __init__(self, path):
        self.ModelDictionary = dict()
        self.BoneDictionary = dict()
        self.Frames = list()

        start = time.time()
        print('Reading T6GR File...')
        t6gr = open(path, "rb")

        self.FileVersion = read_int(t6gr)

        #Change import function depending on file format type
        if self.FileVersion == 0x100:
            main_progressbar = mel.eval("$tmp = $gMainProgressBar")
            print('File Format Version 1.0.0')
            self.ModelCount = read_int(t6gr)

            for i in range(self.ModelCount):
                key = read_uint(t6gr)
                name = read_string(t6gr)
                self.ModelDictionary[key] = name

            self.UniqueBoneCount = read_int(t6gr)
            for i in range(self.UniqueBoneCount):
                id = read_short(t6gr)
                name = read_string(t6gr)
                self.BoneDictionary[id] = name

            self.FPS = read_int(t6gr)
            self.FrameCount = read_int(t6gr)

            cmds.progressBar(main_progressbar, edit=True,
                     beginProgress=True, isInterruptable=False,
                     status='Parsing T6GR File...', maxValue=max(1, self.FrameCount))
            
            #Read All Frames
            for i in range(self.FrameCount):
                cmds.progressBar(main_progressbar, edit=True, step=1)
                #print "Reading File... " + str(round(percent, 2)) + '%'
                currFrame = Frame()

                currFrame.Pov.FOV = read_float(t6gr)
                currFrame.Pov.Origin = read_vector3(t6gr)
                
                currFrame.Pov.Angles = read_vector3(t6gr)

                entityCount = read_int(t6gr)
                #Read All Entities
                for e in range(entityCount):
                    entity = Entity()
                    entity.UniqueIdentifer = read_int(t6gr)
                    entity.flag = read_byte(t6gr)
                    submodelCount = read_int(t6gr)
                    for m in range(submodelCount):
                        submodel = Model()
                        submodel.modelName = read_int(t6gr)
                        submodel.boneCount = read_byte(t6gr)
                        
                        for b in range(submodel.boneCount):
                            bi = BoneInfo()
                            bi.boneID = read_short(t6gr)
                            bi.orient = read_orientation(t6gr)
                            submodel.Bones.append(bi)

                        entity.Submodels.append(submodel)
                    currFrame.Entities.append(entity)
                self.Frames.append(currFrame)
            cmds.progressBar(main_progressbar, edit=True, endProgress=True)
            #Return
        elif self.FileVersion == 0x101:
            print('File Format Version 1.0.1')
        else:
            print('Unsupported File Format. Your import script may be out of date')
        t6gr.close()
        end = time.time()
        print("Finished reading T6GR File in " + str(round(end - start, 2)) + " seconds")


def read_vector3(file):
        vec = Vector3()
        vec.X = read_float(file)
        vec.Y = read_float(file)
        vec.Z = read_float(file)
        return vec

def read_orientation(file):
    orient = orientation_t()
    orient.origin = read_vector3(file)
    orient.axis.ry1 = read_float(file)
    orient.axis.rx1 = read_float(file) * -1
    orient.axis.rz1 = read_float(file)
    orient.axis.ry2 = read_float(file)
    orient.axis.rx2 = read_float(file) * -1
    orient.axis.rz2 = read_float(file)
    orient.axis.ry3 = read_float(file)
    orient.axis.rx3 = read_float(file) * -1
    orient.axis.rz3 = read_float(file)
    return orient

class Vector3:
    X = 0.0
    Y = 0.0
    Z = 0.0

    def __init__(self):
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0

class idMat3:
    ry1=0.0
    rx1=0.0
    rz1=0.0
    ry2=0.0
    rx2=0.0
    rz2=0.0
    ry3=0.0
    rx3=0.0
    rz3=0.0

    def __init__(self):
        self.ry1 - 0.0
        self.rx1 - 0.0
        self.rz1 - 0.0
        self.ry2 - 0.0
        self.rx2 - 0.0
        self.rz2 - 0.0
        self.ry3 - 0.0
        self.rx3 - 0.0
        self.rz3 - 0.0

class orientation_t:
    origin = Vector3()
    axis = idMat3()
    def __init__(self):
        self.origin = Vector3()
        self.axis = idMat3()

class BoneInfo:
    boneID = 0
    orient = orientation_t()
    def __init__(self):
        self.boneID = 0
        self.orient = orientation_t()

class Model:
    modelName = 0
    boneCount = 0
    Bones = list()

    def __init__(self):
        self.modelName = 0
        self.boneCount = 0
        self.Bones = list()

 
class Entity:
    UniqueIdentifer = 0
    flag = 0
    Submodels = list()
    def __init__(self):
        self.UniqueIdentifer = 0
        self.flag = 0
        self.Submodels = list()


class POV:
    FOV = 90.0
    Origin = Vector3()
    Angles = Vector3()
    def __init__(self):
        self.FOV = 90.0
        self.Origin = Vector3()
        self.Angles = Vector3()

class Frame:
    pov = POV()
    Entities = list()
    def __init__(self):
        self.Pov = POV()
        self.Entities = list()



#  _____ __    ____ ____  
# |_   _/ /_  / ___|  _ \ 
#   | || '_ \| |  _| |_) |
#   | || (_) | |_| |  _ < 
#   |_| \___/ \____|_| \_\
                         

 
master = 'T6GR'

try:
    import httplib
except:
    import http.client as httplib

validFPSValues = {	
10: '10fps', 12: '12fps', 15: 'game', 16: '16fps', 20: '20fps', 24: 'film', 25: 'pal', 30: 'ntsc', 40: '40fps', 48: 'show', 50: 'palf', 60: 'ntscf', 75: '75fps', 	
80: '80fps', 100: '100fps', 120: '120fps', 125: '125fps', 150: '150fps', 200: '200fps', 240: '240fps', 250: '250fps', 300: '300fps', 375: '375fps',	
400: '400fps', 500: '500fps', 600: '600fps', 750: '750fps', 1200: '1200fps'}


skeletonNames = {
121108: 't6_player',
95831: 't6_viewarms',
88366: 't6_zombie'
}




#User Inteface Stuff

def have_internet():
    conn = httplib.HTTPConnection("www.pastebin.com", timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False

def __remove_menu__():
    if cmds.control("T6GRMenu", exists=True):
        cmds.deleteUI("T6GRMenu", menu=True)
        
def __importfile_dialog__(filter_str="", caption_str=""):
    if cmds.about(version=True)[:4] == "2012":
        import_from = cmds.fileDialog2(
            fileMode=1, fileFilter=filter_str, caption=caption_str)
    else:
        import_from = cmds.fileDialog2(fileMode=1,
                                       dialogStyle=2,
                                       fileFilter=filter_str,
                                       caption=caption_str)

    if not import_from or import_from[0].strip() == "":
        return None

    path = import_from[0].strip()
    path_split = os.path.splitext(path)
    if path_split[1] == ".*":
        path = path_split

    return path
    
def __importfolder_dialog__(filter_str="", caption_str=""):
    if cmds.about(version=True)[:4] == "2012":
        import_from = cmds.fileDialog2(
            fileMode=3, fileFilter=filter_str, caption=caption_str)
    else:
        import_from = cmds.fileDialog2(fileMode=3,
                                       dialogStyle=2,
                                       fileFilter=filter_str,
                                       caption=caption_str)

    if not import_from or import_from[0].strip() == "":
        return None

    path = import_from[0].strip()
    path_split = os.path.splitext(path)
    if path_split[1] == ".*":
        path = path_split

    return path

def __about_window__():
    cmds.confirmDialog(message="A T6GR file format importer for Maya.\nT6GR only records animation data, and does not export required models or skeletons. These must be exported through the use of Wraith Archon or Greyhound.\n\n- Developed by Airyz\n- Version 1.0.0",
                       button=['OK'], defaultButton='OK', title="About SE Tools")
def __help__():
    if have_internet():
        #read url from a pastebin
        url = 'https://pastebin.com/raw/XeF8XMNn'
        output = urllib2.urlopen(url).read()
        if sys.platform=='win32':
            os.startfile(output)
        print(output)
    else:
        cmds.confirmDialog(message="Internet is not available", button=['OK'], defaultButton='OK', title="Error:")
                       
def __create_menu__():
    __remove_menu__()

    # Create the base menu object
    cmds.setParent(mel.eval("$tmp = $gMainWindow"))
    menu = cmds.menu("T6GRMenu", label="T6GR", tearOff=True)

    cmds.menuItem(label="Import T6GR File", command=lambda x: __import__(
    ), annotation="Imports a T6GR File")
    
    cmds.menuItem(label="Export for UE4", command=lambda x: __ue4Export__(
    ), annotation="Exports each entity seperately as FBX and renames the skeleton")

    cmds.menuItem(label="About", command=lambda x: __about_window__())
    
    cmds.menuItem(label="Help", command=lambda x: __help__(), annotation="Opens a tutorial")



def generateSkeletonHash(skeleton):
    print('GeneratingSkeletonHash')
    hash = 0
    try:
        bones = cmds.listRelatives(skeleton, allDescendents=True, fullPath=False)
        for bone in bones:
            for char in bone:
                hash += ord(char)
    except:
        hash = -1
    return hash


moveViewarmsToOrigin = False

def __ue4Export__():
    errorCount = 0
    export_folder = __importfolder_dialog__("Output folder", "Select folder to output files to...")
    if export_folder == None: return
    main_progressbar = mel.eval("$tmp = $gMainProgressBar")
    
    entities = cmds.listRelatives(master, allDescendents=False, fullPath=False)
    
    for entity in entities:
        #print entity
        objectType = cmds.objectType(entity)
        print(objectType)
        submodels = cmds.listRelatives(entity, allDescendents=False, fullPath=False)
        if moveViewarmsToOrigin:
            if entity == 'viewarms':
            
                for i in range(len(submodels)):
                    model = submodels[i]
                    if '_Joints' in model:
                        bone = cmds.listRelatives(model, allDescendents=False, fullPath=True)[0]
                    #print(bone)
            
                        cmds.cutKey(bone, attribute='translateX')
                        cmds.cutKey(bone, attribute='translateY')
                        cmds.cutKey(bone, attribute='translateZ')
                        cmds.cutKey(bone, attribute='rotateX')
                        cmds.cutKey(bone, attribute='rotateY')
                        cmds.cutKey(bone, attribute='rotateZ')
            
                        cmds.move(0, 0, 0, bone, absolute=True, worldSpace=False)
                        cmds.rotate(0, 0, 0, bone, absolute=True, worldSpace=False)
                    
                        if i > 0:
                            cmds.scale(1, 1, 1, model, absolute=True, worldSpace=True)
        for model in submodels:
            if not '_Joints' in model:
 
                cmds.select(clear=True)
                
                try:
                    cmds.parent(model, world=True)
                    cmds.parent(model + '_Joints', world=True)
                except:
                    errorCount += 1
                    
                skeletonName = model.replace(entity + '_', '') + '_Joints'
                
                skeletonHash = generateSkeletonHash(model + '_Joints')
                
                
                
                if skeletonHash in skeletonNames:
                    skeletonName = skeletonNames[skeletonHash]
                    
                try:
                    cmds.rename(model + '_Joints', skeletonName)
                    print("Skeleton Name = " + skeletonName)
                    print(model + '.fbx')
                except:
                    errorCount += 1
                
                
                
                try:
                    cmds.select('|' + model, add=True)
                    cmds.select('|' + skeletonName, add=True)
                except:
                    errorCount += 1
                
                #Export for fbx
                directory = export_folder + '\\' + entity 
                
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                cmds.file(directory + '\\' + model + '.fbx', exportSelected=True, type='FBX export')
                #break
                
                try:
                    cmds.rename('|' + skeletonName, model + '_Joints')
                    cmds.parent('|' + model, entity)
                    cmds.parent('|' + model + '_Joints', entity)
                    cmds.select(clear=True)
                except:
                    errorCount += 1
                
                
                
    




#Importing Stuff

def importModel(xmodel_folder, modelname, desiredName, rigName):
    filePath = xmodel_folder + '/' + modelname + '/' + modelname + '_LOD0.semodel'
    __load_semodel__(filePath)
    cmds.rename(modelname + '_LOD0', desiredName)
    cmds.rename('Joints', rigName)

def getBone(joints, boneName):
    for bone in joints:
        split = bone.split('|')
        #print(split[len(split)-1])
        if split[len(split)-1] == boneName:
            return bone
    return None






main_progressbar = mel.eval("$tmp = $gMainProgressBar")
prevValue = 0

def __import__():
    keyframeCount = 0
    start = time.time()
    import_file = __importfile_dialog__("T6GR Files (*.t6gr)", "Select T6GR File")
    if import_file == None: return
    
    
    
    import_folder = __importfolder_dialog__("XModels Folder", "Select Folder Containing XModels")
    if import_folder == None: return
    
    print(import_file)
    print(import_folder)
    
    cmds.progressBar(main_progressbar, edit=True,
                     beginProgress=True, isInterruptable=False,
                     status='Parsing T6GR File...', maxValue=max(1, 100))
                     
    T6GR = T6GReader(import_file)
    
    closestFPS = min(validFPSValues, key=lambda x:abs(x-T6GR.FPS))	
    cmds.currentUnit(time=validFPSValues[closestFPS])	
    cmds.playbackOptions(maxTime=len(T6GR.Frames), animationEndTime=len(T6GR.Frames), minTime=0, animationStartTime=0)	
    cmds.currentTime(0) 
    
    
    if not cmds.objExists(master):
        cmds.group(em=True, name=master)
    cmds.progressBar(main_progressbar, edit=True, endProgress=True)
    
    
    print("Importing all required models")
    
    cmds.progressBar(main_progressbar, edit=True,
                     beginProgress=True, isInterruptable=False,
                     status='Importing all models...', maxValue=max(1, len(T6GR.ModelDictionary.values())))
    
    missingModels = ""
    missingModelCount = 0
    
    for model in T6GR.ModelDictionary.values():
        
        filePath = import_folder + '/' + model + '/' + model + '_LOD0.semodel'
        if os.path.isfile(filePath):
            importModel(import_folder, model, model, model + '_Joints')
            groupname = model + '_temp'
            cmds.group(em=True, name=groupname)
            cmds.parent(model, groupname)
            cmds.parent(model +' _Joints', groupname)
            cmds.progressBar(main_progressbar, edit=True, step=1)
            print(model)
        else:
            missingModels += model + "\r\n"
            missingModelCount += 1

    if missingModelCount > 0:
        doContinue = cmds.confirmDialog(message="Error: The following models could not be imported, as the file does not exist: \r\n" + str(missingModels) + "\r\nWould you like to abort the import process?", button=['Yes','No'], defaultButton='Yes', cancelButton='No', dismissString='Yes')
        if doContinue == 'Yes':
            return
        
        
    cmds.progressBar(main_progressbar, edit=True, endProgress=True, status='Importing all models...')
    
    mainCamera = cmds.camera(hfv = 90)
    cameraShape = mainCamera[1]
    cmds.parent(mainCamera[0], 'T6GR')
    cmds.scale(25, 25, 25, mainCamera)
    
    cmds.progressBar(main_progressbar, edit=True,
                     beginProgress=True, isInterruptable=False,
                     status='Importing T6GR...', maxValue=max(1, len(T6GR.Frames)))
                     
    for i in range(len(T6GR.Frames)):
        print("Importing frame: " + str(i) + '/' + str(len(T6GR.Frames)))
        cmds.progressBar(main_progressbar, edit=True, step=1)
        #Move to next frame
        cmds.currentTime(int(i + 1))
        
        Frame = T6GR.Frames[i]

        #Set and keyframe camera
        cmds.rotate(-Frame.Pov.Angles.X, Frame.Pov.Angles.Y - 90, Frame.Pov.Angles.Z, mainCamera, absolute=True, worldSpace=True)
        cmds.move(Frame.Pov.Origin.X,Frame.Pov.Origin.Z,-Frame.Pov.Origin.Y, mainCamera, absolute=True, worldSpace=True)
        cmds.camera(cameraShape, e=True, hfv = Frame.Pov.FOV)
        cmds.setKeyframe(cameraShape, at='focalLength')
        cmds.setKeyframe(mainCamera, at='translateX')
        cmds.setKeyframe(mainCamera, at='translateY')
        cmds.setKeyframe(mainCamera, at='translateZ')
        cmds.setKeyframe(mainCamera, at='rotateX')
        cmds.setKeyframe(mainCamera, at='rotateY')
        cmds.setKeyframe(mainCamera, at='rotateZ')
        keyframeCount += 7
        
        #Import Entities
        for entity in Frame.Entities:
            
            entityName = 'entity_' + str(entity.UniqueIdentifer)
            
            
            if entity.flag == T6GR.ViewmodelFlag:
                entityName='viewarms' #There should only ever be one viewarm entity so we dont need a unique identifier
            if entity.flag == T6GR.CorpseFlag:
                entityName = 'corpse_' + str(entity.UniqueIdentifer)
            if entity.flag == T6GR.Playerflag:
                entityName = 'player_' + str(entity.UniqueIdentifer)
            if entity.flag == T6GR.KillstreakFlag:
                entityName = 'killstreak_' + str(entity.UniqueIdentifer)
            if entity.flag == T6GR.ZombieFlag:
                entityName = 'zombie_' + str(entity.UniqueIdentifer)
            if entity.flag == T6GR.ProjectileFlag:
                entityName = 'projectile_' + str(entity.UniqueIdentifer)
            
            entityName = entityName.replace('-', '')
            #If the entity doesnt exist, create a new group for this entity
            if not cmds.objExists(entityName):
                cmds.group(em=True, name=entityName)
                cmds.parent(entityName, master)
            
            for model in entity.Submodels:
                
                #Convert model int name to string
                modelNameString = T6GR.ModelDictionary[model.modelName]
                importedModelName = entityName + '_' + modelNameString
                importedRigName = entityName +'_' + modelNameString + '_Joints'
                
                #If this model doesn't yet exist in entity, duplicate it from the temp models
                if not cmds.objExists(importedModelName):
                    if cmds.objExists(modelNameString + '_temp'):
                        print('Duplicating Model')
                        cmds.duplicate(modelNameString + '_temp', un=True, name=importedModelName)
                        armature = importedModelName + '|' + modelNameString + '_Joints'
                        cmds.rotate(0,0,0, armature, absolute=True, worldSpace=True)
                        cmds.scale(0.394, 0.394, 0.394, armature, absolute=True, worldSpace=True)
                        cmds.rename(armature, entityName + '_' + modelNameString + '_Joints')
                        cmds.parent(entityName + '_' + modelNameString + '_Joints', entityName)
                        cmds.parent(importedModelName, entityName)
                
                
                errorCount = 0
                try:
                    joints =  cmds.listRelatives(entityName + '_' + modelNameString + '_Joints', allDescendents=True, fullPath=True)
                    for BoneInfo in model.Bones:
                        boneName = T6GR.BoneDictionary[BoneInfo.boneID]
                        
                        orient = BoneInfo.orient
                    
                        #Convert from matrix to maya coordinates
                        boneRotX = math.degrees(math.atan2(orient.axis.rz2, orient.axis.rz3))
                        boneRotY = math.degrees(math.atan2(-orient.axis.rz1, math.sqrt(orient.axis.rz2**2+orient.axis.rz3**2)))
                        boneRotZ = math.degrees(math.atan2(orient.axis.ry1, orient.axis.rx1))
                    
                        #Find bone object inside entity
                        bone = getBone(joints, boneName)
                        
                        if (orient.origin.X == orient.origin.Y == orient.origin.Z == 0) == False:
                            
                            #Position Bone
                            cmds.move(orient.origin.X, orient.origin.Y, orient.origin.Z, bone, absolute=True, worldSpace=True)
                            cmds.rotate(boneRotX, boneRotY, boneRotZ - 90, bone, absolute=True, worldSpace=True)
                            
                            #Keyframe Bone
                            cmds.setKeyframe(bone, at='translateX')
                            cmds.setKeyframe(bone, at='translateY')
                            cmds.setKeyframe(bone, at='translateZ')
                            cmds.setKeyframe(bone, at='rotateX')
                            cmds.setKeyframe(bone, at='rotateY')
                            cmds.setKeyframe(bone, at='rotateZ')
                            keyframeCount += 6
                except:
                    errorCount += 1
                        
    cmds.progressBar(main_progressbar, edit=True, endProgress=True)
    #Clean up the temp models
    print('Cleaning Up...')
    cmds.rename(mainCamera[0], 'POV')
    for key in T6GR.ModelDictionary:
        try:
            tempModel = T6GR.ModelDictionary[key] + '_temp'
            cmds.delete(tempModel)
        except:
            print("could not find model to delete")
    
    print('Rotating entities to match maya coordinate space...')
    entities = cmds.listRelatives(master, allDescendents=False, fullPath=False)
    for entity in entities:
        children = cmds.listRelatives(entity, allDescendents=False, fullPath=True)
        for child in children:
            if '_Joints' in child:
                cmds.rotate(-90, 0, 0, child, absolute=True, worldSpace=True)
    
    end = time.time()
    print('Successfully imported ' + str(keyframeCount) + ' keyframes')
    print('Finished importing T6GR in ' + str(round(end - start, 2)) + ' seconds')
                    
__create_menu__()


def initializePlugin(m_object):
    __create_menu__()
    
    
def uninitializePlugin(m_object):
    __remove_menu__()
