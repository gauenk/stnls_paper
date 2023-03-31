"""

Output for denoising table

"""

import numpy as np
import pandas as pd
from easydict import EasyDict as edict

def run(records):
    fields = ['psnrs','ssims']
    fmts = ["%2.3f","%2.3f"]
    fmts = {field:fmt for field,fmt in zip(fields,fmts)}
    for dname,ddf in records.groupby("dname"):
        for arch,adf in ddf.groupby("arch_name"):
            for sigma,sdf in adf.groupby("sigma"):
                for vid_name,vdf in sdf.groupby("vid_name"):
                    for wt,wdf in vdf.groupby("wt"):
                        mes = "(%s,%s,%d,%d,%s): " % (dname,arch,sigma,wt,vid_name)
                        for field in fields:
                            mf = wdf[field].apply(np.mean).mean()
                            mes += fmts[field] % mf + " "
                        uuid = wdf['uuid'].to_numpy()[0]
                        full = "[%s]: %s" % (uuid,mes)
                        print(full)


def create_header(labels):
    head = """
    \\begin{table*}[h]
    \centering
    \\resizebox{\linewidth}{!}{%
    \\begin{tabular}{crrrrrrrrr}
        \\toprule
        \multicolumn{1}{c}{}\n"""
    indent8 = " "*8

    # -- labeled cols --
    for label in labels:
        head += indent8 + "& \multicolumn{2}{c}{%s}\n" % label
    head = head[:-1]
    head += "\\\\\n"

    # -- crules --
    j = 2
    for ip in range(len(labels)):
        i0,i1 = j,j+2
        head += indent8 + "\cmidrule(l){%d-%d}\n"  % (i0,i1)
        j = i1+1
    head += indent8 + "\multicolumn{1}{c}{$\sigma$}\n"

    # -- mulicols --
    chunk = """
        & \multicolumn{1}{c}{Original}
        & \multicolumn{1}{c}{Ours}\n"""
    for label in labels:
        head += indent8 + "%% --- %s --- %%" % label
        head += chunk
    # head = head[:-1]
    head += indent8 + "\\\\\n"
    return head

def label_origin(records):
    # origin = np.array(["Original" for _ in range(len(records))])
    origin = np.where(records['wt'] == 3,"Ours","Original")
    records['origin'] = origin

def run_latex(records,fields,fields_summ,res_fields,res_fmt):


    # -- format records --
    label_origin(records)

    sigma_order = {"15":0,"30":1,"50":2}
    arch_order = {"colanet":0,"lidia":1,"n3net":2}
    records = records.sort_values("sigma",key=lambda x: x.map(sigma_order))
    records = records.sort_values("arch_name",key=lambda x: x.map(arch_order))
    records.reset_index(inplace=True,drop=True)

    # -- init report --
    report = ""
    grouped = records
    order = ["COLA-Net","LIDIA","N3Net"]
    report = create_header(order)
    # print(grouped['arch_name'].unique())

    # -- create table --
    indent8 = " "*8
    for sigma,sdf in grouped.groupby("sigma"):
        row = indent8 + "%d\n" % sigma
        for arch,adf in sdf.groupby("arch_name"):
            # print(sigma,arch) # check order
            row += indent8
            order = ["psnrs","ssims","strred"]
            fmts = ["%2.2f","%1.3f","%2.1f"]
            ours = edict({f:-1 for f in order})
            orig = edict({f:-1 for f in order})
            for group,gdf in adf.groupby("origin"):
                for field in order:
                    val = float(gdf[field].apply(np.mean).mean())
                    if field == "strred":
                        if np.isnan(val): val = -1.
                        else: val = val*100.
                    if group == "Ours":
                        ours[field] = val
                    elif group == "Original":
                        orig[field] = val
                    else:
                        raise ValueError(f"Uknown group [{group}]")
            diffs = edict()
            for key in order:
                diffs[key] = ours[key] - orig[key]
            args = [orig[k] for k in order]
            args += [ours[k] for k in order]
            row_i = "& %2.2f/%1.3f/%2.1f & "
            for f,k in zip(fmts,order):
                d_s = f % diffs[k]
                if k == "strred":
                    if diffs[k] < 0: row_i += "\\textbf{" + f + "}/"
                    else: row_i += f + "/"
                else:
                    if diffs[k] > 0: row_i += "\\textbf{" + f + "}/"
                    else: row_i += f + "/"
            row_i = row_i[:-1] + "\n"
            row_i = row_i % tuple(args)
            row += (row_i % list(args))
        row += indent8 + "\\\\\n"
        report += row

    # -- tail of table --
    report += indent8 + "\\bottomrule\n"
    report += " "*4 + "\end{tabular}\n"
    report += " "*4 + "}\n"
    report += " "*4 + "\\caption{}\n"
    report += " "*4 + "\end{table*}\n"

    # -- remove first 4 spaces --
    final_report = ""
    for line in report.split("\n"):
        final_report += line[4:] + "\n"
    return final_report

