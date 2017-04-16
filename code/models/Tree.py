class Tree(object):
    def __init__(self, data=None, auxilaries=None):
        self.children = []
        self.data = data
        self.parent = None;
        self.auxilaries = auxilaries;
    def travers(self):
        datas = [];
        if self.data is not None:
            datas.append(self.data)
        for c in self.children:
            datas.extend(c.travers());
        return datas;
    def find(self, data):
        if self.data == data:
            return self;
        else:
            for c in self.children:
                found = c.find(data);
                if found is not None:
                    return found;
            return None;
    def addChild(self, data, auxilaries=None):
        child = Tree(data=data, auxilaries=auxilaries);
        self.addChildNode(child);
        return child;
    def addChildNode(self, child):
        child.setParent(self);
        self.children.append(child);
        return child;
    def setParent(self, parent):
        self.parent = parent;
        return self, parent;
    def getParent(self):
        return self.parent;
    def getRoot(self):
        if self.getParent() is None:
            return self;
        else:
            return self.getParent().getRoot();
    def printTree(self):
        levels = self.extractLevelsWithInfo();
        for level in sorted(levels.iterkeys()):
            points = levels[level];
            leftIntend = level*5;
            for point in points:
                node  = point[0];
                spaceToPrint = point[1];
                # print ''.center(leftIntend-1), str(node).center(5), ''.center(spaceToPrint-5-2),;
                print '(', node, ',', node.parent, ')',
            print '\n';
    def extractLevelsWithInfo(self, level=0, levels={}):
        spaceToPrint = self.getNeededSpaceToPrint();
        if level in levels.keys():
            thisLevel = levels[level];
            thisLevel.append((self, spaceToPrint));
        else:
            thisLevel = [];
            thisLevel.append((self, spaceToPrint));
            levels[level] = thisLevel;
        for c in self.children:
            levels = c.extractLevelsWithInfo(level=level+1, levels=levels);
        return levels;
    def getNeededSpaceToPrint(self):
        space = 5;
        for c in self.children:
            space += c.getNeededSpaceToPrint();
        return space;
    def extractThreads(self):
        threads = [];
        if len(self.children) > 0:
            for c in self.children:
                childThreads = c.extractThreads();
                for thread in childThreads:
                    if self.data is not None:
                        thread.insert(0, self);
                threads.extend(childThreads);
            return threads;
        else:
            threads = [[self]];
            return threads;
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return self.__str__()