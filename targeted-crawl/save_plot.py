from tldextract import extract
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from other_strategies import dumb, randomCrawl, balanced

######################################################################################
def SavePlot(params, env, saveDirPlots, epoch, sset, arrRL, totReward, totDiscountedReward):
    crawlLen = min(params.maxDocs, len(env.nodes))
    arrDumb = dumb(env, crawlLen, params)
    arrRandom = randomCrawl(env, crawlLen, params)
    arrBalanced = balanced(env, crawlLen, params)
    #print("arrRL", len(arrRL))

    avgRandom = 0
    avgBalanced = 0
    avgRL = 0
    
    url = env.rootURL
    domain = extract(url).domain

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(arrDumb, label="dumb ", color='black')
    ax.plot(arrRandom, label="random {0:.1f}".format(avgRandom), color='firebrick')
    ax.plot(arrBalanced, label="balanced {0:.1f}".format(avgBalanced), color='red')
    ax.plot(arrRL, label="RL {0:.1f} {1:.1f} {2:.1f}".format(avgRL, totReward, totDiscountedReward), color='salmon')

    ax.legend(loc='upper left')
    plt.xlabel('#crawled')
    plt.ylabel('#found')
    plt.title("{sset} {domain}".format(sset=sset, domain=domain))

    fig.savefig("{dir}/{sset}-{domain}-{epoch}.png".format(dir=saveDirPlots, sset=sset, domain=domain, epoch=epoch))
    plt.close()
