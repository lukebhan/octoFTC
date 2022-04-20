params= ["mode", "save_path", "trajectories", "model", "simulations", "save_rew", "save_pos", "save_torques", "iterations", "fault_random", "fault_mag", "fault_motor","training_algo"]
metadata = {"mode": {"type": "string", "values": ["test", "train"]}, 
            "save_path": {"type": "stringOpenVal"}, 
            "trajectories": {"type": "string", "values": ["all", "zigzag", "Etraj", "circle"]}, 
            "model": {"type": "string", "values": ["true", "false"]},
            "simulations": {"type": "int"},
            "save_rew": {"type": "string", "values": ["true", "false"]},
            "save_pos": {"type": "string", "values": ["true", "false"]},
            "save_torques": {"type": "string", "values": ["true", "false"]},
            "fault_mag": {"type": "float"},
            "iterations": {"type": "int"},
            "fault_random": {"type": "string", "values": ["true", "false"]},
            "fault_motor": {"type": "intBounded", "values": [1, 2, 3, 4, 5, 6, 7, 8]
                },
            "training_algo": {"type": "string", "values": ["ppo", "sac"]}}


def parser(filename):
    parameterDict = {}
    lineNum = 0
    with open(filename) as f:
        line = f.readline()
        lineNum+=1
        # skip all header comments
        while(line != ""):
            line, lineNum = getNextLine(f, lineNum) 
            parseLine(line, lineNum, parameterDict)
    return parameterDict

def parseLine(line, lineNum, parameterDict):
    line = line.split("#")[0]
    line = line.split("=")
    param = line[0].strip()
    if(param in params):
        if(metadata[param]["type"] == "string"):
            if(line[1].strip() in metadata[param]["values"]):
                parameterDict[param] = line[1].strip()
            else:
                invalidFile(lineNum)
        elif(metadata[param]["type"] == "stringOpenVal"):
            parameterDict[param] = line[1].strip()
        elif(metadata[param]["type"] == "int"):
            try:
                val = int(line[1].strip())
                parameterDict[param] = val
            except:
                invalidFile(lineNum)
        elif(metadata[param]["type"] == "float"):
            try:
                val = float(line[1].strip())
                parameterDict[param] = val
            except:
                invalidFile(lineNum)
        elif(metadata[param]["type"] == "intBounded"):
            try:
                val = int(line[1].strip())
                if(val in metadata[param]["values"]):
                    parameterDict[param] = val
            except:
                invalidFile(lineNum)
        else:
            invalidFile(lineNum)
    return 
            

def invalidFile(num):
   raise Exception("File formatting is invalid near line", num)

def getNextLine(f, lineNum):
    line = f.readline()
    lineNum += 1
    while(line != "" and( line[0] == "#" or line == "\n")):
        line = f.readline()
        lineNum += 1
    return line, lineNum
