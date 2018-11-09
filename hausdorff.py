# -*- coding: utf-8 -*-
if __name__ == '__main__':

    import os
    import numpy as np
    from scipy.spatial.distance import directed_hausdorff


    def erase_newline(s):
        return s.strip()

    def read_text(filename):
        with open(filename) as f:
            lines = list(map(erase_newline, f.readlines()))
        return lines

    test_basename = read_text("./intersection_path_prediction/data/test_basenames.txt")
    GT_PATH = "./intersection_path_prediction/tracking"
    RESULT_PATH = "./RESULT_Obs_ex15_tr15000_te15000"

    hausdorff_list = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []

    for bn in test_basename:
        gt_coord = np.loadtxt(os.path.join(GT_PATH, bn + ".txt"), dtype=np.int32)
        result_coord = np.loadtxt(os.path.join(RESULT_PATH, bn + "-path.txt"), dtype=np.int32)
        # print ("%s >> %0.3f" % (bn, max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0])))
        hausdorff_list.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))
        if "scene1" in bn:
            list1.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))
        elif "scene2" in bn:
            list2.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))
        elif "scene3" in bn:
            list3.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))
        elif "scene4" in bn:
            list4.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))
        elif "scene5" in bn:
            list5.append(max(directed_hausdorff(result_coord, gt_coord)[0], directed_hausdorff(gt_coord, result_coord)[0]))

    print "scene1\n", list1
    print ("平均 >> %0.3f" % np.average(list1))
    print ("標準偏差 >> %0.3f" % np.std(list1))
    print "=" * 40
    print "scene2\n", list2
    print ("平均 >> %0.3f" % np.average(list2))
    print ("標準偏差 >> %0.3f" % np.std(list2))
    print "=" * 40
    print "scene3\n", list3
    print ("平均 >> %0.3f" % np.average(list3))
    print ("標準偏差 >> %0.3f" % np.std(list3))
    print "=" * 40
    print "scene4\n", list4
    print ("平均 >> %0.3f" % np.average(list4))
    print ("標準偏差 >> %0.3f" % np.std(list4))
    print "=" * 40
    print "scene5\n", list5
    print ("平均 >> %0.3f" % np.average(list5))
    print ("標準偏差 >> %0.3f" % np.std(list5))

    print "=" * 40
    print "all\n", hausdorff_list
    print ("平均 >> %0.3f" % np.average(hausdorff_list))
    print ("標準偏差 >> %0.3f" % np.std(hausdorff_list))
