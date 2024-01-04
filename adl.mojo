#attempt to run the python code on the new mojo language

from python import Python

fn main() raises:
    Python.add_to_path("/Users/vasudevmenon/ADL/cw/CW-ADL-ee20947")

    # import all the modules
    let dataload = Python.import_module("dataload")
    let main = Python.import_module("main")
    let misc = Python.import_module("misc")
    let models = Python.import_module("models")
    let train = Python.import_module("train")

    try:
        dataload.main()
    catch let e:
        print("Error in dataload:", e)

    try:
        main.main()
    catch let e:
        print("Error in main:", e)

    try:
        misc.main()
    catch let e:
        print("Error in misc:", e)

    try:
        models.main()
    catch let e:
        print("Error in models:", e)

    try:
        train.main()
    catch let e:
        print("Error in train:", e)
