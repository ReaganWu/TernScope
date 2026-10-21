"""Scalar independent input-vector addressing specification, no cycle prediction.
reader(node_name, y, x, channel) returns signed producer A8.
"""
def array_input(n, output_y, output_x, k, shapes, reader, luts):
    if k >= n['K']: return 0
    ci=n['ci']; kh,kw=n['kernel']; sy,sx=n['stride']; ph,pw=n['padding']
    spatial,ch=divmod(k,ci); ky,kx=divmod(spatial,kw)
    iy=output_y*sy+ky-ph; ix=output_x*sx+kx-pw
    s0=n['sources'][0]; _,h,w=shapes[s0['node']];h<<=s0['ry'];w<<=s0['rx']
    if not(0<=iy<h and 0<=ix<w):return 0 # padding after edge transform
    for s in n['sources']:
        lo=s['start']*16; count=min(s['count']*16,ci-lo)
        if lo<=ch<lo+count:
            v=reader(s['node'],iy>>s['ry'],ix>>s['rx'],s['offset']*16+ch-lo)
            return int(luts[s['lut']][v&255])
    raise AssertionError('unmapped consumer channel')

def producer_row(y,x,c,h,w,storage_c):
    assert storage_c%16==0
    return (y*w+x)*(storage_c//16)+c//16,c%16
