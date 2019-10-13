"""Wrappers for SIRIUS DFT_ground_state class"""


class DFT_ground_state:
    def __init__(self, dft_obj):
        self.dft_obj


    def update(self):
        """"""
        # TODO extrapolate density

        self.dft_obj.update()
