from .base_options import BaseOptions

class DecompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('decomp_image', type=str, help='path of image to decompose')
        self.isTrain = False
