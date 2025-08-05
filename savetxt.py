def savetxt(targetdir, name, data):
    with open(targetdir + name + '.txt', 'w') as fp:
        for item in data:
            # write each item on a new line
            fp.write("%s\n" % item)