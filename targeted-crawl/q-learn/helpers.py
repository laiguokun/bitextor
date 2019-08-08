import numpy as np
import hashlib
import sys

from common import Timer

global TIMER
TIMER = Timer()


class Transition:
    def __init__(self, currURLId, nextURLId, done, features, siblings, numNodes, numURLs, targetQ):
        self.currURLId = currURLId
        self.nextURLId = nextURLId 
        self.done = done
        self.features = features 
        self.siblings = siblings
        self.numNodes = numNodes
        self.numURLs = numURLs
        self.targetQ = targetQ

    def DebugTransition(self):
        ret = str(self.currURLId) + "->" + str(self.nextURLId)
        return ret


class Candidates:
    def __init__(self):
        self.dict = {} # nodeid -> link
        self._urlIds = []

        self.dict[0] = []
        self._urlIds.append(0)

    def AddLink(self, link):
        urlLId = link.childNode.urlId
        if urlLId not in self.dict:
            self.dict[urlLId] = []
            self._urlIds.append(link.childNode.urlId)
        self.dict[urlLId].append(link)

    def RemoveLink(self, nextURLId):
        del self.dict[nextURLId]
        self._urlIds.remove(nextURLId)

    def copy(self):
        ret = Candidates()
        ret.dict = self.dict.copy()
        ret._urlIds = self._urlIds.copy()

        return ret

    def AddLinks(self, env, urlId, visited, params):
        currNode = env.nodes[urlId]
        #print("   currNode", curr, currNode.Debug())
        newLinks = currNode.GetLinks(visited, params)

        for link in newLinks:
            self.AddLink(link)

    def _GetNextURLIds(self, params):
        ret = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        urlIds = self._urlIds[1:]
        #random.shuffle(urlIds)
        i = 1
        for urlId in urlIds:
            ret[0, i] = urlId
            i += 1
            if i >= params.NUM_ACTIONS:
                break

        return ret, i

    def GetFeaturesNP(self, env, params, visited):
        urlIds, numURLs = self._GetNextURLIds(params)
        #print("urlIds", urlIds)

        langFeatures = np.zeros([params.NUM_ACTIONS, params.FEATURES_PER_ACTION], dtype=np.int)
        siblings = np.zeros([1, params.NUM_ACTIONS], dtype=np.int)

        i = 0
        for urlId in urlIds[0]:
            #print("urlId", urlId)
            
            links = self.dict[urlId]
            if len(links) > 0:
                link = links[0]
                #print("link", link.parentNode.urlId, link.childNode.urlId, link.text, link.textLang)
                langFeatures[i, 0] = link.textLang

                parentNode = link.parentNode
                #print("parentNode", childId, parentNode.lang, parentLangId, parentNode.Debug())
                langFeatures[i, 1] = parentNode.lang

                matchedSiblings = self.GetMatchedSiblings(env, urlId, parentNode, visited)
                numMatchedSiblings = len(matchedSiblings)
                #if numMatchedSiblings > 1:
                #    print("matchedSiblings", urlId, parentNode.urlId, matchedSiblings, visited)
                
                siblings[0, i] = numMatchedSiblings
                
            i += 1
            if i >= numURLs:
                #print("overloaded", len(self.dict), self.dict)
                break

        #print("BEFORE", ret)
        langFeatures = langFeatures.reshape([1, params.NUM_ACTIONS * params.FEATURES_PER_ACTION])
        #print("AFTER", ret)
        #print()
        numNodes = np.empty([1,1])
        numNodes[0,0] = len(visited)
        
        numURLsRet = np.empty([1,1])
        numURLsRet[0,0] = numURLs

        return urlIds, numURLsRet, langFeatures, siblings, numNodes

    def GetMatchedSiblings(self, env, urlId, parentNode, visited):
        ret = []

        #print("parentNode", urlId)
        for link in parentNode.links:
            sibling = link.childNode
            if sibling.urlId != urlId:
                #print("   link", sibling.urlId, sibling.alignedDoc)
                if sibling.urlId in visited:
                    # sibling has been crawled
                    if sibling.alignedNode is not None and sibling.alignedNode.urlId in visited:
                        # sibling has been matched
                        ret.append(sibling.urlId)      

        return ret


class Link:
    def __init__(self, text, textLang, parentNode, childNode):
        self.text = text 
        self.textLang = textLang 
        self.parentNode = parentNode
        self.childNode = childNode


def StrNone(arg):
    if arg is None:
        return "None"
    else:
        return str(arg)


def NormalizeURL(url):
    url = url.lower()
    ind = url.find("#")
    if ind >= 0:
        url = url[:ind]
        #print("pageURL", pageURL)
    #if url[-5:] == ".html":
    #    url = url[:-5] + ".htm"
    #if url[-9:] == "index.htm":
    #    url = url[:-9]
    if url[-1:] == "/":
        url = url[:-1]

    if url[:7] == "http://":
        #print("   strip protocol1", url, url[7:])
        url = url[7:]
    elif url[:8] == "https://":
        #print("   strip protocol2", url, url[8:])
        url = url[8:]

    return url


class Node:
    def __init__(self, urlId, url, docIds, langIds, redirectId):
        assert(len(docIds) == len(langIds))
        self.urlId = urlId
        self.url = url
        self.docIds = set(docIds)

        self.redirectId = redirectId
        self.redirect = None

        self.links = set()
        self.lang = 0 if len(langIds) == 0 else langIds[0]
        self.alignedNode = None

        self.normURL = None

        #print("self.lang", self.lang, langIds, urlId, url, docIds)
        #for lang in langIds:
        #    assert(self.lang == lang)

    def CreateLink(self, text, textLang, childNode):            
        link = Link(text, textLang, self, childNode)
        self.links.add(link)

    def GetLinks(self, visited, params):
        ret = []
        for link in self.links:
            childNode = link.childNode
            childURLId = childNode.urlId
            #print("   ", childNode.Debug())
            if childURLId != self.urlId and childURLId not in visited:
                ret.append(link)
        #print("   childIds", childIds)

        return ret

    def Recombine(self, loserNode):
        assert (loserNode is not None)
        # print("Recombining")
        # print("   ", self.Debug())
        # print("   ", loserNode.Debug())

        self.docIds.update(loserNode.docIds)
        self.links.update(loserNode.links)

        if self.lang == 0:
            if loserNode.lang != 0:
                self.lang = loserNode.lang
        else:
            if loserNode.lang != 0:
                assert (self.lang == loserNode.lang)

        if self.alignedNode is None:
            if loserNode.alignedNode is not None:
                self.alignedNode = loserNode.alignedNode
        else:
            if loserNode.alignedNode is not None:
                print(self.alignedNode.Debug())
                print(loserNode.alignedNode.Debug())
                assert (self.alignedNode == loserNode.alignedNode)

    def Debug(self):
        return " ".join([str(self.urlId), self.url, StrNone(self.docIds),
                        StrNone(self.lang), StrNone(self.alignedNode),
                        StrNone(self.redirect), str(len(self.links)),
                        StrNone(self.normURL) ] )


class Env:
    def __init__(self, sqlconn, url):
        self.rootURL = url
        self.numAligned = 0
        self.nodes = {} # urlId -> Node
        self.url2urlId = {}
        self.maxLangId = 0

        unvisited = {} # urlId -> Node
        visited = {} # urlId -> Node
        rootURLId = self.Url2UrlId(sqlconn, url)
        self.rootNode = self.CreateNode(sqlconn, visited, unvisited, rootURLId, url)
        self.CreateGraphFromDB(sqlconn, visited, unvisited)
        print("CreateGraphFromDB", len(visited))
        #for node in visited.values():
        #    print(node.Debug())

        self.ImportURLAlign(sqlconn, visited)

        #print("rootNode", rootNode.Debug())
        print("Recombine")
        normURL2Node = {}
        self.Recombine(visited, normURL2Node)
        
        self.rootNode = normURL2Node[self.rootNode.normURL]
        assert(self.rootNode is not None)
        print("rootNode", self.rootNode.Debug())

        self.PruneNodes(self.rootNode)

        startNode = Node(sys.maxsize, "START", [], [], None)
        startNode.CreateLink("", 0, self.rootNode)
        self.nodes[startNode.urlId] = startNode

        # stop node
        node = Node(0, "STOP", [], [], None)
        self.nodes[0] = node

        self.UpdateStats()
        print("self.nodes", len(self.nodes), self.numAligned, self.maxLangId)
        #for node in self.nodes.values():
        #    print(node.Debug())

        print("graph created")

    def ImportURLAlign(self, sqlconn, visited):
        #print("visited", visited.keys())
        sql = "SELECT id, url1, url2 FROM url_align"
        val = ()
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        for res in ress:
            urlId1 = res[1]
            urlId2 = res[2]
            #print("urlId", urlId1, urlId2)

            #print("   ", urlId1, urlId2)
            if urlId1 not in visited or urlId2 not in visited:
                print("Alignment not in graph", urlId1, urlId2)
                continue

            node1 = visited[urlId1]
            node2 = visited[urlId2]
            node1.alignedNode = node2
            node2.alignedNode = node1

    def UpdateStats(self):
        for node in self.nodes.values():
            if node.alignedNode is not None:
                self.numAligned += 1

            if node.lang > self.maxLangId:
                self.maxLangId = node.lang

            for link in node.links:
                if link.textLang > self.maxLangId:
                    self.maxLangId = link.textLang

    def PruneNodes(self, rootNode):
        visit = []
        visit.append(rootNode)

        while len(visit) > 0:
            node = visit.pop()
            self.nodes[node.urlId] = node

            # prune links to non-docs
            linksCopy = set(node.links)
            for link in linksCopy:
                childNode = link.childNode
                if len(childNode.docIds) == 0:
                    #print("empty", childNode.Debug())
                    node.links.remove(link)
                elif childNode.urlId not in self.nodes:
                    visit.append(childNode)

    def GetRedirectedNormURL(self, node):
        while node.redirect is not None:
            node =  node.redirect
        normURL = NormalizeURL(node.url)
        return normURL

    def Recombine(self, visited, normURL2Node):
        #print("visited", visited.keys())
        # create winning node for each norm url
        for node in visited.values():
            node.normURL = self.GetRedirectedNormURL(node)
            if node.normURL not in normURL2Node:
                normURL2Node[node.normURL] = node
            else:
                winner = normURL2Node[node.normURL]
                winner.Recombine(node)

        # relink aligned nodes & child nodes to winning nodes
        for node in visited.values():
            if node.alignedNode is not None:
                newAlignedNode = normURL2Node[node.alignedNode.normURL]
                node.alignedNode = newAlignedNode

            for link in node.links:
                childNode = link.childNode
                #print("childNode", childNode.Debug())
                newChildNode = normURL2Node[childNode.normURL]
                #print("newChildNode", newChildNode.Debug())
                #print()
                link.childNode = newChildNode

    def CreateNode(self, sqlconn, visited, unvisited, urlId, url):
        if urlId in visited:
            return visited[urlId]
        elif urlId in unvisited:
            return unvisited[urlId]
        else:
            docIds, langIds, redirectId = self.UrlId2Responses(sqlconn, urlId)
            node = Node(urlId, url, docIds, langIds, redirectId)
            assert(urlId not in visited)
            assert(urlId not in unvisited)
            unvisited[urlId] = node
            return node

    def CreateGraphFromDB(self, sqlconn, visited, unvisited):
        while len(unvisited) > 0:
            (urlId, node) = unvisited.popitem()
            visited[node.urlId] = node
            #print("node", node.Debug())
            assert(node.urlId == urlId)

            if node.redirectId is not None:
                assert(len(node.docIds) == 0)
                redirectURL = self.UrlId2Url(sqlconn, node.redirectId)
                redirectNode = self.CreateNode(sqlconn, visited, unvisited, node.redirectId, redirectURL)
                node.redirect = redirectNode
            else:
                linksStruct = self.DocIds2Links(sqlconn, node.docIds)

                for linkStruct in linksStruct:
                    childURLId = linkStruct[0]
                    childUrl = self.UrlId2Url(sqlconn, childURLId)
                    childNode = self.CreateNode(sqlconn, visited, unvisited, childURLId, childUrl)
                    link = Link(linkStruct[1], linkStruct[2], node, childNode)
                    node.links.add(link)

            #print("   ", node.Debug())
            
    def DocIds2Links(self, sqlconn, docIds):
        docIdsStr = ""
        for docId in docIds:
            docIdsStr += str(docId) + ","

        sql = "SELECT id, url_id, text, text_lang_id FROM link WHERE document_id IN (%s)"
        val = (docIdsStr,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        linksStruct = []
        for res in ress:
            struct = (res[1], res[2], res[3])
            linksStruct.append(struct)

        return linksStruct

    def UrlId2Responses(self, sqlconn, urlId):
        sql = "SELECT id, status_code, to_url_id, lang_id FROM response WHERE url_id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        ress = sqlconn.mycursor.fetchall()
        assert (ress is not None)

        docIds = []
        langIds = []
        redirectId = None
        for res in ress:
            if res[1] == 200:
                assert(redirectId == None)
                docIds.append(res[0])
                langIds.append(res[3])
            elif res[1] in (301, 302):
                assert(len(docIds) == 0)
                redirectId = res[2]

        return docIds, langIds, redirectId

    def RespId2URL(self, sqlconn, respId):
        sql = "SELECT T1.id, T1.val FROM url T1, response T2 " \
            + "WHERE T1.id = T2.url_id AND T2.id = %s"
        val = (respId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0], res[1]


    def UrlId2Url(self, sqlconn, urlId):
        sql = "SELECT val FROM url WHERE id = %s"
        val = (urlId,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    def Url2UrlId(self, sqlconn, url):
        #print("url",url)
        c = hashlib.md5()
        c.update(url.lower().encode())
        hashURL = c.hexdigest()

        if hashURL in self.url2urlId:
            return self.url2urlId[hashURL]

        sql = "SELECT id FROM url WHERE md5 = %s"
        val = (hashURL,)
        sqlconn.mycursor.execute(sql, val)
        res = sqlconn.mycursor.fetchone()
        assert (res is not None)

        return res[0]

    ########################################################################
    def GetNextState(self, params, action, visited, urlIds):
        assert(urlIds.shape[1] > action)
        nextURLId = urlIds[0, action]
        #print("   nextNodeId", nextNodeId)
        nextNode = self.nodes[nextURLId]
        if nextURLId == 0:
            #print("   stop")
            reward = 0.0
        elif nextNode.alignedNode is not None and nextNode.alignedNode.urlId in visited:
            reward = params.reward
            #print("   visited", visited)
            #print("   nodeIds", nodeIds)
            #print("   reward", reward)
            #print()
        else:
            #print("   non-rewarding")
            reward = params.cost

        return nextURLId, reward

    def Walk(self, start, params, sess, qn, printQ):
        visited = set()
        unvisited = Candidates()
        
        curr = start
        i = 0
        numAligned = 0
        totReward = 0.0
        totDiscountedReward = 0.0
        discount = 1.0
        mainStr = str(curr) + "->"
        rewardStr = "  ->"
        debugStr = ""

        while True:
            # print("curr", curr)
            # print("hh", next, hh)
            currNode = self.nodes[curr]
            unvisited.AddLinks(self, currNode.urlId, visited, params)
            urlIds, numURLs, featuresNP, siblings, numNodes = unvisited.GetFeaturesNP(self, params, visited)
            #print("featuresNP", featuresNP)
            #print("siblings", siblings)

            if printQ: 
                numURLsScalar = int(numURLs[0,0])
                urlIdsTruncate = urlIds[0, 0:numURLsScalar]                
                unvisitedStr =  str(urlIdsTruncate)

            action, allQ = qn.Predict(sess, featuresNP, siblings, numNodes, numURLs)
            nextURLId, reward = self.GetNextState(params, action, visited, urlIds)
            totReward += reward
            totDiscountedReward += discount * reward
            visited.add(nextURLId)
            unvisited.RemoveLink(nextURLId)

            alignedStr = ""
            nextNode = self.nodes[nextURLId]
            if nextNode.alignedNode is not None:
                alignedStr = "*"
                numAligned += 1

            if printQ:
                debugStr += "   " + str(curr) + "->" + str(nextURLId) + " " \
                         + str(action) + " " \
                         + str(numURLsScalar) + " " \
                         + unvisitedStr + " " \
                         + str(allQ) + " " \
                         + str(featuresNP) + " " \
                         + "\n"

            #print("(" + str(action) + ")", str(nextURLId) + "(" + str(reward) + ") -> ", end="")
            mainStr += str(nextURLId) + alignedStr + "->"
            rewardStr += str(reward) + "->"
            curr = nextURLId
            discount *= params.gamma
            i += 1

            if nextURLId == 0: break

        mainStr += " " + str(i) + "/" + str(numAligned)
        rewardStr += " " + str(totReward) + "/" + str(totDiscountedReward)

        if printQ:
            print(debugStr, end="")
        print(mainStr)
        print(rewardStr)

        return numAligned, totReward, totDiscountedReward



    def WalkAll(self, params, sess, qn):
        for node in self.nodes.values():
            self.Walk(node.urlId, params, sess, qn, False)

    def GetNumberAligned(self, path):
        ret = 0
        for transition in path:
            next = transition.nextURLId
            nextNode = self.nodes[next]
            if nextNode.alignedDoc > 0:
                ret += 1
        return ret


    def Neural(self, epoch, currURLId, params, sess, qnA, qnB, visited, unvisited, docsVisited):
        TIMER.Start("Neural.1")
        #DEBUG = False

        unvisited.AddLinks(self, currURLId, visited, params)
        urlIds, numURLs, featuresNP, siblings, numNodes = unvisited.GetFeaturesNP(self, params, visited)
        #print("   childIds", childIds, unvisited)
        TIMER.Pause("Neural.1")

        TIMER.Start("Neural.2")
        action, Qs = qnA.Predict(sess, featuresNP, siblings, numNodes, numURLs)
        if np.random.rand(1) < params.eps:
            #if DEBUG: print("   random")
            action = np.random.randint(0, params.NUM_ACTIONS)
        TIMER.Pause("Neural.2")
        
        TIMER.Start("Neural.3")
        nextURLId, r = self.GetNextState(params, action, visited, urlIds)
        nextNode = self.nodes[nextURLId]
        #if DEBUG: print("   action", action, next, Qs)
        TIMER.Pause("Neural.3")

        TIMER.Start("Neural.4")
        visited.add(nextURLId)
        unvisited.RemoveLink(nextURLId)
        nextUnvisited = unvisited.copy()
        TIMER.Pause("Neural.4")

        TIMER.Start("Neural.5")
        if nextURLId == 0:
            done = True
            maxNextQ = 0.0
        else:
            assert(nextURLId != 0)
            done = False

            # Obtain the Q' values by feeding the new state through our network
            nextUnvisited.AddLinks(self, nextNode.urlId, visited, params)
            _, nextNumURLs, nextFeaturesNP, nextSiblings, nextNumNodes = nextUnvisited.GetFeaturesNP(self, params, visited)
            nextAction, nextQs = qnA.Predict(sess, nextFeaturesNP, nextSiblings, nextNumNodes, nextNumURLs)        
            #print("nextNumNodes", numNodes, nextNumNodes)
            #print("  nextAction", nextAction, nextQ)

            #assert(qnB == None)
            #maxNextQ = np.max(nextQs)

            _, nextQsB = qnB.Predict(sess, nextFeaturesNP, nextSiblings, nextNumNodes, nextNumURLs)
            maxNextQ = nextQsB[0, nextAction]
        TIMER.Pause("Neural.5")
            
        TIMER.Start("Neural.6")
        targetQ = Qs
        #targetQ = np.array(Qs, copy=True)
        #print("  targetQ", targetQ)
        newVal = r + params.gamma * maxNextQ
        targetQ[0, action] = (1 - params.alpha) * targetQ[0, action] + params.alpha * newVal
        #targetQ[0, action] = newVal
        self.ZeroOutStop(targetQ, urlIds, numURLs, params.unusedActionCost)

        #if DEBUG: print("   nextStates", nextStates)
        #if DEBUG: print("   targetQ", targetQ)

        transition = Transition(currURLId, 
                                nextNode.urlId, 
                                done, 
                                np.array(featuresNP, copy=True), 
                                np.array(siblings, copy=True), 
                                numNodes,
                                numURLs,
                                np.array(targetQ, copy=True))
        TIMER.Pause("Neural.6")

        return transition

    def Trajectory(self, epoch, currURLId, params, sess, qns):
        visited = set()
        unvisited = Candidates()
        docsVisited = set()

        while (True):
            tmp = np.random.rand(1)
            if tmp > 0.5:
                qnA = qns.q[0]
                qnB = qns.q[1]
            else:
                qnA = qns.q[1]
                qnB = qns.q[0]
            #qnA = qns.q[0]
            #qnB = None

            transition = self.Neural(epoch, currURLId, params, sess, qnA, qnB, visited, unvisited, docsVisited)
            
            qnA.corpus.AddTransition(transition, params.deleteDuplicateTransitions)

            currURLId = transition.nextURLId
            #print("visited", visited)

            if transition.done: break
        #print("unvisited", unvisited)
        
    def ZeroOutStop(self, targetQ, urlIds, numURLs, unusedActionCost):
        #print("urlIds", numURLs, targetQ, urlIds)
        assert(targetQ.shape == urlIds.shape)
        targetQ[0,0] = 0.0
        
        #i = 0
        #for i in range(urlIds.shape[1]):
        #    if urlIds[0, i] == 0:
        #        targetQ[0, i] = 0

        numURLsScalar = int(numURLs[0,0])
        for i in range(numURLsScalar, targetQ.shape[1]):
            targetQ[0, i] = unusedActionCost

        #print("targetQ", targetQ)
