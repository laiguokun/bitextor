import os
import sys
import numpy as np

from other_strategies import dumb, randomCrawl, balanced

######################################################################################
def SavePlots(sess, qns, params, envs, saveDirPlots, epoch, sset):
    for env in envs:
        SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset)

######################################################################################
def SavePlot(sess, qns, params, env, saveDirPlots, epoch, sset):
    print("Walking", env.rootURL)
    arrDumb = dumb(env, len(env.nodes), params)
    arrRandom = randomCrawl(env, len(env.nodes), params)
    arrBalanced = balanced(env, len(env.nodes), params)
    arrRL, totReward, totDiscountedReward = Walk(env, params, sess, qns)

    url = env.rootURL
    domain = extract(url).domain

    warmUp = 200
    avgRandom = avgBalanced = avgRL = 0.0
    for i in range(len(arrDumb)):
        if i > warmUp and arrDumb[i] > 0:
            avgRandom += arrRandom[i] / arrDumb[i]
            avgBalanced += arrBalanced[i] / arrDumb[i]
            avgRL += arrRL[i] / arrDumb[i]

    avgRandom = avgRandom / (len(arrDumb) - warmUp)
    avgBalanced = avgBalanced / (len(arrDumb) - warmUp)
    avgRL = avgRL / (len(arrDumb) - warmUp)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(arrDumb, label="dumb ", color='maroon')
    ax.plot(arrRandom, label="random {0:.1f}".format(avgRandom), color='firebrick')
    ax.plot(arrBalanced, label="balanced {0:.1f}".format(avgBalanced), color='red')
    ax.plot(arrRL, label="RL {0:.1f} {1:.1f}".format(avgRL, totDiscountedReward), color='salmon')

    ax.legend(loc='upper left')
    plt.xlabel('#crawled')
    plt.ylabel('#found')
    plt.title("{sset} {domain}".format(sset=sset, domain=domain))

    fig.savefig("{dir}/{sset}-{domain}-{epoch}.png".format(dir=saveDirPlots, sset=sset, domain=domain, epoch=epoch))
    fig.show()

