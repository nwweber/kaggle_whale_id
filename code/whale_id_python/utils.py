__author__ = 'niklas'

import os


pathjoin = os.path.join

def guess_location():
    # easiest if environment variable is defined
    if 'whale_id_location' in os.environ:
        return os.environ['whale_id_location']

    # otherwise check for known hostname
    hostname_mapping = {"coma": "cluster", "luna-l": "laptop"}
    import socket
    hostname = socket.gethostname()
    if hostname in hostname_mapping.keys():
        return hostname_mapping[hostname]

    # otherwise try to discriminate based on paths
    if os.path.isdir(pathjoin("/", "scratch")):
        return "cluster"
    else:
        return "laptop"
