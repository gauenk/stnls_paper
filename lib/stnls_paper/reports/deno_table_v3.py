"""

Output for denoising table

"""

import numpy as np
import pandas as pd
from easydict import EasyDict as edict


def run_latex(records):

    # -- format records --
    sigma_order = {"15":0,"30":1,"50":2}
    # arch_order = {"colanet":0,"lidia":1,"n3net":2}
    records = records.sort_values("sigma",key=lambda x: x.map(sigma_order))
    # records = records.sort_values("python_module",key=lambda x: x.map(arch_order))
    records.reset_index(inplace=True,drop=True)

    # -- init report --
    report = ""
    grouped = records
    # order = ["COLA-Net","LIDIA","N3Net"]
    # report = create_header(order)
    # print(grouped['arch_name'].unique())
    order = ["psnrs","ssims","strred"]


    # -- create table --
    indent8 = " "*2
    for sigma,sdf in grouped.groupby("sigma"):
        row = indent8 + "%d\n" % sigma
        gkeys = ["0_False","3_False","3_True"]
        res = edict({g:{f:-1 for f in order} for g in gkeys})
        for group,adf in sdf.groupby(["wt","read_flows"]):
            # print(sigma,arch) # check order
            row += indent8
            fmts = ["%2.2f","%1.3f","%2.1f"]
            res_key = "%d_%s" % group

            # orig = edict({f:-1 for f in order})
            # for group,gdf in adf.groupby("origin"):
            for field in order:
                val = float(adf[field].apply(np.mean).mean())
                if field == "strred":
                    if np.isnan(val): val = -1.
                    else: val = val*100.
                res[res_key][field] = val
                # if group == "Ours":
                #     ours[field] = val
                # elif group == "Original":
                #     orig[field] = val
                # else:
                #     raise ValueError(f"Uknown group [{group}]")
            # print(res)
            # diffs = edict()
            # for key in order:
            #     diffs[key] = ours[key] - orig[key]
            args = [res[res_key][k] for k in order]
            # args += [ours[k] for k in order]
            row_i = "& %2.2f/%1.3f/%2.1f & "
            # for f,k in zip(fmts,order):
            #     d_s = f % diffs[k]
            #     if k == "strred":
            #         if diffs[k] < 0: row_i += "\\textbf{" + f + "}/"
            #         else: row_i += f + "/"
            #     else:
            #         if diffs[k] > 0: row_i += "\\textbf{" + f + "}/"
            #         else: row_i += f + "/"
            # row_i = row_i[:-1] + "\n"
            row_i = row_i % tuple(args)
            row += (row_i % list(args))
        row += indent8 + "\\\\\n"
        report += row

    # -- tail of table --
    # report += indent8 + "\\bottomrule\n"
    # report += " "*4 + "\end{tabular}\n"
    # report += " "*4 + "}\n"
    # report += " "*4 + "\\caption{}\n"
    # report += " "*4 + "\end{table*}\n"

    # -- remove first 4 spaces --
    final_report = ""
    for line in report.split("\n"):
        final_report += line[2:] + "\n"
    return final_report

