import os

#The function I used for renaming files correctly
def replace(parent): #to replace whitespaces with undescores in file and dir names
    for path, folders, files in os.walk(parent):
        for i in range(len(folders)):
            new_name = folders[i].replace(' ', '_')
            os.rename(os.path.join(path, folders[i]), os.path.join(path, new_name))
            folders[i] = new_name


        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', '_')))