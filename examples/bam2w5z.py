import gdata
import sys
import tempfile

with tempfile.NamedTemporaryFile(delete=True) as positive:
    with tempfile.NamedTemporaryFile(delete=True) as negative:
        positive = positive.name
        negative = negative.name
        gdata.utils.bam_cov(sys.argv[1], positive, output2=negative, stranded=True)
        gdata.utils.bw_to_w5z(positive, sys.argv[2])
        gdata.utils.bw_to_w5z(negative, sys.argv[3])
