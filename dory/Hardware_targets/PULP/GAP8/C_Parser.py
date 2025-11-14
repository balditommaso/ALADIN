

from dory.Hardware_targets.PULP.Common import C_Parser_PULP

import os

class C_Parser(C_Parser_PULP):

    def __init__(self, *args, **kwargs):
        super(C_Parser, self).__init__(*args, **kwargs)
        if self.precision_library == "mixed-hw":
            assert False, "optional='mixed-hw' not compatible with GAP8!"


    def get_file_path(self):
        return "/".join(os.path.realpath(__file__).split("/")[:-1])
