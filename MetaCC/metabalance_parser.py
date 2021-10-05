import os
file_prefix = "CCmb_384983_" # old way
folder = './cmllogs/'
files = [filename for filename in os.listdir(folder) if filename.startswith(file_prefix)]
files.sort()
print(files)
print(len(files))

dict = {}
final_roc_best = 0
best_about = ''

for f in files:

    run = folder + f
    run = open(run, 'r')
    lines = run.read().splitlines()
    run.close()
    print()
    print(f)
    if len(lines) == 12:
        about = lines[1].split(',')
        #print(about)
        for info in about:
            infos = info.split('=')
        #    print(infos)

        lr_info = float(about[4].split('=')[1])
        #print(lr_info)

        inner_sampling_info = about[3].split('=')[1]
        #print(inner_sampling_info)

        outer_sampling_info = about[10].split('=')[1]
        #print(outer_sampling_info)

        val = lines[-1].split(' ')
        print(lines[-2])
        final_roc = float(val[0])
        max_roc = float(val[1])
        k = inner_sampling_info + " " + outer_sampling_info
        if k not in dict.keys():
            dict[k] = []
        print(inner_sampling_info, outer_sampling_info, lr_info, final_roc, max_roc)
        dict[k].append((lr_info, final_roc, max_roc))
        # if lr_info not in dict.keys():
        #     dict[lr_info] = {}
        # dict[lr_info][inner_sampling_info + "_" + outer_sampling_info] = [final_roc, max_roc]
        if final_roc > final_roc_best:
            final_roc_best = final_roc
            best_about = (inner_sampling_info, outer_sampling_info, lr_info, final_roc, max_roc)
    # break

print(best_about)
for k in dict.keys():
    print('###################################')
    print(k)
    all = dict[k]
    for l in all:
        print(l)