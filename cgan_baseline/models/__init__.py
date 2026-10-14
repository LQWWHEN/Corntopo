from .CPTP_model import CPTPModel
from .mCPTP_model import mCPTPModel


def create_model(args, stats):
    return {'CPTP': CPTPModel, 'mCPTP': mCPTPModel}[args.model](args, stats)
