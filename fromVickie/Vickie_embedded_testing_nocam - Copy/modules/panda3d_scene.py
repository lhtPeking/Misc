from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from panda3d.core import *
import numpy as np
from scipy import interpolate

class Panda3D_Scene(ShowBase):
    def __init__(self, shared):

        self.shared = shared
        # The line with undecorated was not there in the file copied from engert lab. It works for now so leaving it here
        loadPrcFileData("",
                        """fullscreen 0
                           win-origin 1920 0
                           undecorated 1
                           win-size 2160 1920
                           sync-video 1
                           load-display pandagl""")

        ShowBase.__init__(self)

        self.disableMouse()
        self.lens = PerspectiveLens()
        self.lens.setFov(90, 90)
        self.lens.setNearFar(0.001, 1000)
        self.lens.setAspectRatio(2160./1920)

        self.cam.node().setLens(self.lens)

        self.setBackgroundColor(0, 0, 0, 1)

        # Create the four fishnodes (positions where cards or other objects are anchored per fish)
        self.fish_nodes = [self.render.attachNewNode("fish_node%d" % i ) for i in range(4)]

    def create_circles(self, n, edges=20):

        vdata = GeomVertexData('name', GeomVertexFormat.getV3t2(), Geom.UHStatic)

        prim_wall = GeomTriangles(Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, 'vertex')
        texcoord_writer = GeomVertexWriter(vdata, 'texcoord')

        angs = np.linspace(0, 360, edges)

        for i in range(n):
            for j in range(len(angs)):
                ang0 = angs[j]
                ang1 = angs[(j + 1) % edges]
                vertex_writer.addData3f(0, i, 0)  # stack them in distance
                vertex_writer.addData3f(np.cos(ang0 * np.pi / 180), i, np.sin(ang0 * np.pi / 180))
                vertex_writer.addData3f(np.cos(ang1 * np.pi / 180), i, np.sin(ang1 * np.pi / 180))

                texcoord_writer.addData2f(i / float(n), 0)
                texcoord_writer.addData2f(i / float(n), 0.5)
                texcoord_writer.addData2f(i / float(n), 1)

            prim_wall.addConsecutiveVertices(i * edges * 3, edges * 3)
            prim_wall.closePrimitive()

        geom_wall = Geom(vdata)
        geom_wall.addPrimitive(prim_wall)

        circles = GeomNode('card')
        circles.addGeom(geom_wall)

        return NodePath(circles)

    def make_cardnodes(self):

        # Place cards on the fishnodes
        self.cardnodes = []
        cm = CardMaker('card')

        for i in range(4):
            self.cardnodes.append(self.fish_nodes[i].attachNewNode(cm.generate()))

        for i in range(4):
            self.cardnodes[i].setScale(2, 1, 2)
            self.cardnodes[i].setPos(-1, 0, -1)

        # Place dummy textures on the cards

        dummypic = (np.zeros((1024, 1024, 4))).astype(np.uint8)
        dummytex = Texture("texture")

        dummytex.setMagfilter(Texture.FTLinear)
        dummytex.setMinfilter(Texture.FTLinearMipmapLinear)
        dummytex.setAnisotropicDegree(16)

        dummytex.setup2dTexture(dummypic.shape[1], dummypic.shape[0], Texture.TUnsignedByte, Texture.FRgba32)

        memoryview(dummytex.modify_ram_image())[:] = dummypic.tobytes()

        for i in range(4):
            self.cardnodes[i].setTexture(dummytex)
