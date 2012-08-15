#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re, sqlite3, nltk, subprocess
from collections import Counter, deque
import scipy.spatial.distance as sdist
import Pycluster as pc
from time import time

def zscore_distr(d):
	# Produit une distribution z-score ∈[0;1]
	a=np.array(d)
	r=a-a.mean()
	r=r/r.std()
	return r
	
def human_time(t):
	csecs=(t-int(t))*100
	mins, secs = divmod(int(t), 60)
	hours, mins = divmod(mins, 60)
	
	r = '%02d:%02d:%02d.%02d' % (hours, mins, secs, csecs)
	r = re.sub("^[0:]*","",r)
	r = re.sub("[0]*$","",r)
	r = re.sub("\.$","",r)
	
	if ":" not in r: r=r+" secondes"
	
	return r

def grepsearch(pattern, string):
	# Plus rapide que regexp de Python, et prend probablement moins de mémoire.
	p1=subprocess.Popen(["echo", string], stdout=subprocess.PIPE)
	p2=subprocess.Popen(["grep", "-E",  pattern], stdin=p1.stdout, stdout=subprocess.PIPE)
	p1.stdout.close()
	return p2.communicate()[0]
	

def grepisin(pattern, string):
	# Plus rapide que regexp de Python, et prend probablement moins de mémoire.
	p1=subprocess.Popen(["echo", string], stdout=subprocess.PIPE)
	p2=subprocess.Popen(["grep", "-E",  pattern], stdin=p1.stdout, stdout=subprocess.PIPE)
	p1.stdout.close()
	return p2.wait()==0

def Burrows_delta(mat):
	# Distance utilisée en stylométrie
	# Cf. J. F. Burrows, “Delta: a measure of stylistic difference and a guide to likely authorship,” Literary and Linguistic Computing 17, pp. 267–287, 2002.
	# mat: matrice des fréquences de mot 
	
	if isinstance(mat,np.ndarray):
		m=mat
	else:
		m=np.array(mat)
		
	r=np.zeros((mat.shape[0],mat.shape[0]))
	sig=m.std(0)
	for i in range(m.shape[0]):
		for j in range(i, m.shape[0]):
			rr=(abs(m[i]-m[j])/sig).sum()
			r[i][j]=rr
			r[j][i]=rr
	return r

class docDB:
	#Base de donnée des documents. Génère des matrices à partir de ça.
	nonLetter=re.compile(u"[0-9:;,.’()[\]*&?%$#@!~|\\\/=+_¬}{¬¤¢£°±\n\r‘’“”«—·–»…¡¿̀`~^><'\"]")
	tokenizer=nltk.tokenize.WordPunctTokenizer()
	fdist=None
	
	def __init__(self, truc, table, typ="", colTexte="Texte", colID="rowid", lang="french", stemmer=None, filterStopWords=False, filterDocHapax=False, stem=False, lettersOnly=True, lemmatize=False, maxOccur=0.4, minOccur=0.03):
		#On peut initialiser à partir de données d'une db sqlite. Davantage à venir.
		
		if isinstance(truc, sqlite3.Connection) and table != "":
			# truc est un objet db sqlite
			self.db=truc
			self.table=table
		
		elif typ=="path":
			# "truc" est une chaîne contenant le chemin vers une db sqlite
			self.db=sqlite3.connect(truc)
			self.table=table
		
		else:
			self.db=sqlite3.connect(":memory:")
			self.table="documents"
		
		self.curs=self.db.cursor()
		self.curs.execute("pragma table_info(%s)" % self.table)
		self.cols=[ i[1] for i in self.curs.fetchall() ]
		
		self.colTexte=colTexte
		self.colID=colID
		self.lang=lang
		
		self.filterStopWords=filterStopWords
		self.stem=stem
		self.lettersOnly=lettersOnly
		self.lemmatize=lemmatize
		self.filterDocHapax=filterDocHapax
		self.maxOccur=maxOccur
		self.minOccur=minOccur
		
		if stemmer==None:
			self.stemmer=nltk.stem.SnowballStemmer(lang)
		else: self.stemmer=stemmer
	
	def _ajouter_variables(self, mat):
		# Ajouter les autres variables
		lsvals=list(set(self.cols[:]+[self.colID]))
		lsvals.remove(self.colTexte)
		
		self.curs.execute("select {0} from {1}".format(", ".join(lsvals), self.table))
		vars(mat).update(dict(zip(lsvals,np.array(self.curs.fetchall()).T)))
		
		return mat
	
	def _preTraitement(self, t, filtrerDocHapax=None):
		# Tokenize et filtre un texte t, retourne liste de mots
		# Fait pour être lancé avant une concordance
		# Filtres: miniscule, enlever les non-lettres, stopwords
		
		if isinstance(t,list) or isinstance(t, np.ndarray) or isinstance(t, deque):
			r=" ".join(list(t))
		else: r=t
		
		# Mettre en minuscule
		r=r.lower()
		
		# Filtres
		if self.lettersOnly:
			r=re.sub(self.nonLetter,"",r)
		if self.filterStopWords:
			swre=" *\\b("+"|".join(nltk.corpus.stopwords.words(self.lang))+")\\b *"
			r=re.sub(swre," ", r)
			
		return self.tokenizer.tokenize(r)
	
	def _postTraitement(self, listeMots):
		# Filtre une liste de mots
		# Fait pour être lancé après une concordance
		# Filtres: mots trop ou trop peu fréquents
		
		l=listeMots[:]
		
		#Stem
		if self.stem:
			l = [ [ self.stemmer.stem(mot) for mot in ln ] for ln in l ]
		
		# Lemmatize
		#English-only (wordnet)
		if self.lemmatize:
			lem=nltk.wordnet.WordNetLemmatizer()
			l = [ [ lem.lemmatize(mot) for mot in ln ] for ln in l ]
		
		# Filtrer selon fréquence
		
		# faire la fréquence des mots
		import itertools
		wf=Counter(list(itertools.chain(*l)))
		
		# calculer occurrence maximum et minimum
		if isinstance(self.maxOccur,float) and self.maxOccur<=1.0:
			maxo=self.maxOccur*len(l)
		else: maxo = int(self.maxOccur)
		
		if isinstance(self.minOccur,float) and self.minOccur>=0.0:
			mino=int(np.ceil(self.minOccur*len(l)))
		else: mino = self.minOccur
		
		# Do it
		l=[ [ i for i in ln if wf[i]>=mino and wf[i]<=maxo ] for ln in l ]
		
		return l, wf.keys()
	
	def filtrer(self, t, typ="list", filtrerDocHapax=None):
		# Tokenize et filtre un texte t, retourne liste de mots (sauf si typ=="str")
		
		# Mettre en minuscule
		r=t.lower()
		
		# Filtres
		if self.lettersOnly:
			r=re.sub(self.nonLetter,"",r)
		if self.filterStopWords:
			swre=" *\\b("+"|".join(nltk.corpus.stopwords.words(self.lang))+")\\b *"
			r=re.sub(swre," ", r)
			
		tl=self.tokenizer.tokenize(r)
		if self.stem:
			tl = [ self.stemmer.stem(i) for i in tl ]
		elif self.lemmatize:
			lem=nltk.wordnet.WordNetLemmatizer()
			tl = [ lem.lemmatize(i) for i in tl ]
		
		# Enlever les mots n'apparaissant qu'une fois
		if (self.filterDocHapax and not filtrerDocHapax==False) or filtrerDocHapax==True:
			fd=nltk.probability.FreqDist(i for i in tl)
			for i in fd.hapaxes():
				tl.remove(i)
		
		# Retour
		if typ=="str":
			return " ".join(tl)
		elif typ=="list":
			return tl
		else: return tl
	
	def MakeFDist(self):
		# Faire la distribution des fréquences des mots pour tout le corpus
		self.fdist=nltk.probability.FreqDist()
		self.curs.execute("select {0} from {1}".format(self.colTexte,self.table))
		for ln in self.curs.fetchall():
			for i in self.filtrer(ln[0]):
				self.fdist.inc(i.lower())
	
	def MakeWFMat(self, hapax=False, maxMots=5000):
		# Faire la matrice — pour les opérations mathématiques
		
		self.MakeFDist()
		
		# Liste de mots
		motls=[ i.encode('utf8') for i in self.fdist.keys()[:maxMots] ]
		motids=np.array(tuple(range(len(motls))), dtype=zip(motls, ["<i4"]*len(motls)))
		
		# Rappel des textes de la DB sqlite
		self.curs.execute("select {0} from {1}".format(self.colTexte, self.table) )
		mat=deque()
		
		# Faire la matrice elle-même
		for ln in self.curs.fetchall():
			matln=np.zeros(len(motls),dtype="<i4")
			
			wfcounter=Counter(self.filtrer(ln[0]))  # Compte occurrence de chaque mot
			wfkeys=[i.encode('utf8') for i in wfcounter.keys()]
			wfarray=np.array(tuple(wfcounter.values()), dtype=zip(wfkeys, ["<i4"]*len(wfcounter)))
			motscommuns=list(set(wfkeys) & set(motls)) # Ne prend que les mots qui sont dans le top 5000 (maxMots) pour le corpus
			matln[list(motids[motscommuns])]=list(wfarray[motscommuns]) # Ajoute les valeurs
			
			mat.append(matln)
		mat=WFMat(mat)
		
		#Ajouter la liste de mots
		mat.mots=motls[:]
		
		# Ajouter les autres variables et envoyer la sauce
		return self._ajouter_variables(mat)
		
	def concordance(self, mot, maxMots=5000, fenetre = 50, postStem = False, regex=True):
		# Fait une concordance et retourne une matrice
		
		# Chercher tous les articles dont le texte contiennent le mot
		lscontextes=[]
		if isinstance(mot,unicode):
			mo=self.filtrer(mot)[0].encode("utf8")
		else: mo=mot
		
		if regex:
			regex=lambda x,y: grepregex(x,y) != ''
			self.db.create_function("REGEXP", 2, regex)
			#~ self.db.enable_load_extension(True)
			#~ self.db.load_extension('/usr/lib/sqlite3/pcre.so')
			vmatch = np.vectorize(lambda x: grepisin(mo, x))
			self.curs.execute("select Texte from articles where Texte REGEXP '{0}'".format(mo))
		else:
			vmatch = np.vectorize(lambda x: x==mot)
			self.curs.execute("select Texte from articles where Texte like '%{0}%'".format(mo))

		for ln in self.curs.fetchall():
			tl=np.array([ i.encode('utf8') for i in self._preTraitement(ln[0])])
			for i in np.arange(len(tl))[vmatch(tl)]:
				lb = i-fenetre if i>=fenetre else 0
				ub = i+fenetre
				lscontextes.append(tl[lb:ub])
		
		# Faire fréquence de mot globale
		lscontextes, motls=self._postTraitement(lscontextes)
		motids=np.array(tuple(range(len(motls))), dtype=zip(motls, ["<i4"]*len(motls)))
		
		# Faire la matrice elle-même
		mat=deque()
		
		for ln in lscontextes:
			matln=np.zeros(len(motls),dtype="<i4")
			
			wfcounter=Counter(ln)  # Compte occurrence de chaque mot
			wfkeys=wfcounter.keys()
			wfarray=np.array(tuple(wfcounter.values()), dtype=zip(wfkeys, ["<i4"]*len(wfcounter)))
			motscommuns=list(set(wfkeys) & set(motls)) # Ne prend que les mots qui sont dans le top 5000 (maxMots) pour le corpus
			matln[list(motids[motscommuns])]=list(wfarray[motscommuns]) # Ajoute les valeurs
			
			mat.append(matln)
		
		wfm=WFMat(mat, self)
		
		#Ajouter la liste de mots
		wfm.mots=motls[:]
		
		# Ajouter les autres variables et envoyer la sauce
		return self._ajouter_variables(wfm)

class WFMat:
	# Matrice de fréquence de mots
	mots=[]
	
	def __init__(self, mat, db=None):
		self.mat=np.array(mat)
			
	def distance(self, metric="jaccard"):
		# Produit une matrice de distance entre documents
		if metric=="burrows":
			return distance(Burrows_delta(self.mat), mat=self)
		else:
			return distance(sdist.squareform(sdist.pdist(self.mat,metric=metric)), mat=self)
	
	def multikmeans(self, krange=None):
		# La recette magique
		
		if krange==None:
			kr=np.arange(2, len(self.mat)-1)
		else: kr=krange
		lmat=len(self.mat)
		
		accords=np.zeros((lmat,lmat), dtype=int) # Où on comptera combien de fois chq paire de documents est classé ensemble
		t=deque() # pour sauver temps & mémoire, on emploie deque à la place de list
		t0=time()
		k2s = lambda x: x*0.85
		tunits=k2s(np.array(kr)).sum()
		
		# La boucle elle-même
		for k in kr:
			t1=time()
			
			# K-means
			c,err,nfound=pc.kcluster(self.mat,k)
			
			# Mise à jour des valeurs
			for i in np.unique(c):
				accords[c==i] += c==i
			
			# Prédiction du temps restant
			t2=time()
			tunits-=k2s(k)
			t.append((t2-t1)/k2s(k))
			prediction = tunits*np.mean(tuple(t)[-20:])
			print "k={0}: \t{1} ({2} depuis le début) \t{3} à faire".format(k,human_time(t2-t1),human_time(t2-t0),human_time(prediction))
		
		return accords/float(k)

class distance:
	# Matrice de distance entre documents
	graph=None
	pos=None
	
	def __init__(self, distmat, mat=None):
		self.d=np.array(distmat)
		self.zd=zscore_distr(self.d)
		self.mat=mat
	
	def mkGraph(self):
		# fait un graphe NetworkX à partir des distances. Surtout pour les projections Fruchteman-Reingold
		self.graph=nx.Graph()
		for i in xrange(len(self.zd)):
			for j in xrange(i, len(self.zd)):
				self.graph.add_edge(i,j,weight=1-self.zd[i][j])
	
	def projFruchtemanReingold(self, partition=None, sortie=None, positions=None):
		# Projette sur 2 dimensions grâces à un algorithme de force
		
		if self.graph==None: self.mkGraph()
		
		plt.clf()
		if positions!=None:
			self.pos=positions
		elif self.pos==None:	
			self.pos=nx.spring_layout(self.graph, weighted=True)
		
		dvals={"with_labels": False, "node_size":10}
		if partition!=None:
			dvals["node_color"]=partition[:]
			dvals["vmin"]=min(partition)
			dvals["vmax"]=max(partition)
		nx.draw_networkx_nodes(self.graph, self.pos, **dvals)
		
		if sortie==None:
			plt.show()
		else:
			plt.savefig(savefile)
	
	def cluster_kmedoids(self, k=2, npass=50):
		# Utilise la distance pour produire une partition de k classes
		# n est le nombre d'itérations
		
		c, err, nfound = pc.kmedoids(self.zd, k, npass=npass)
		
		return partition(c, self.mat)
		
class partition:
	# Partition des documents
	classes=[]
	idf=None
	
	def __init__(self, clusterIDs, mat):
		self.codes=list(clusterIDs)
		self.mat=mat
		self.mkClasses()
	
	def __len__(self):
		return len(self.classes)
		
	def __str__(self):
		r=""
		for c in self.classes:
			r+=str(c)+"\n\n"
	
	def mkClasses(self):
		# Faire les classes à partir de la liste des numéros de classe pour chq document
		l=np.arange(len(self.codes))
		for i in np.unique(self.codes):
			self.classes.append(classe(l[np.array(self.codes)==i],self))
	
	def calcIDF(self):
		# Retourne l'IDF, considérant une classe comme un document
		self.idf = np.log(len(self.codes)/np.array([ c.freqs.clip(0,1) for c in self.classes ]).sum(0))
	
	def calcTFIDF(self):
		self.calcIDF()
		for c in self.classes:
			c.calcTFIDF()
			
class classe(list):
	tfidf=None
	
	def __init__(self, docids, partition):
		self[:]=docids
		self.partition=partition
		self.freqs=partition.mat.mat[docids].sum(0)
	
	def __str__(self):		
		return self.prettyprint()
			
	def prettyprint(self, valeurs=None, n=5):
		# Retourne un coefficient d'association dans un format adapté à un terminal
		if valeurs==None:
			if self.tfidf==None:
				self.calcTFIDF()
			vals=self.tfidf
		else: vals=valeurs
		
		ls=sorted(zip(vals,self.partition.mat.mots), reverse=True)[:5]
		r="Classe {0} (N={1})".format(ls[0][1], len(self))
		hline="+" + "-" * (len(r)+2) + "+"
		r="%s\n| %s |\n%s\n" % (hline, r, hline)
		for val, mot in ls:
			r+="\t{0}\t{1}\n".format(mot, val)
		
		return r
	
	def calcTFIDF(self):
		if self.partition.idf==None: self.parition.calcIDF()
		self.tfidf = self.freqs/float(self.freqs.sum())*self.partition.idf
	
	def signeAssociationChi2(self):
		# Positif si la présence d'un mot est associée à la classe, négatif si c'est son absence qui l'est
		m=self.partition.mat.mat
		E=len(self)*m.sum(0)/float(len(m))
		O=m[self].sum(0)
		
		return (O<E)*2 - 1
	
	def motsChi2(self):
		# Donne une liste de mots avec les mots sous-représentés entre parenthèse
		l=np.array(self.partition.mat.mots)
		sousreps=self.signeAssociationChi2()
		l[sousreps==-1]=["(%s)" % i for i in l[sousreps==-1]]
		return l
		
	def calcChi2(self):
		# Calcule le coefficient d'association χ² pour tous les mots
		m=self.partition.mat.mat
		nmat=(m*(-1))+1
		
		# Calcule la valeur absolue du coefficient χ²
		G=numpy.zeros(len(self.partition.codes))
		G[self]=1
		nG=(G*(-1))+1
		N=m.sum(0)
		N10=(m.T*nG).T.sum(0)
		N01=(nmat.T*G).T.sum(0)
		N11=(m.T*G).T.sum(0)
		N00=(nmat.T*nG).T.sum(0)
		
		self.chi2=(N*(N11*N00-N10*N01)**2)/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))
		for i in xrange(len(N11)):
			if N11[i]==N00[i]==0 or N11[i]==N00[i]==1:
				self.chi2[i]=0.0
		
		# Ajoute le signe négatif pour signifier une association négative
		self.chi2=self.chi2*self.signeAssociationChi2()
