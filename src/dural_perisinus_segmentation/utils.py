from __future__ import annotations

import json

import click
from clinicadl.io.bids import BidsFileType


class BidsFileTypeParam(click.ParamType[BidsFileType]):
    def convert(self, value, param, ctx) -> BidsFileType:
        try:
            dict_ = json.loads(value)
            return BidsFileType(**dict_)
        except ValueError:
            self.fail(f"{value!r} is not a valid integer", param, ctx)
