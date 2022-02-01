interval = '15m'
first_date = '01 Apr 2017'
last_date = 'now UTC'
candles_request = 1000
requests_minute = 1200
zip_pack =1
#symbols = ['ETHUSDT', 'BTCUSDT']
# USDT
symbols = ['ETHUSDT', 'BNBUSDT', 'BCCUSDT', 'NEOUSDT', 'LTCUSDT', 'QTUMUSDT', 'ADAUSDT', 'XRPUSDT', 'ETHTUSD',
           'TUSDETH', 'TUSDBNB', 'EOSUSDT', 'TUSDUSDT', 'IOTAUSDT', 'XLMUSDT', 'ONTUSDT', 'TRXUSDT', 'ETCUSDT',
           'ICXUSDT', 'VENUSDT', 'NULSUSDT', 'VETUSDT', 'PAXUSDT', 'USDCBNB', 'BCHABCUSDT', 'BCHSVUSDT', 'BNBTUSD',
           'XRPTUSD', 'EOSTUSD', 'XLMTUSD', 'BNBUSDC', 'ETHUSDC', 'XRPUSDC', 'EOSUSDC', 'XLMUSDC', 'USDCUSDT',
           'ADATUSD', 'TRXTUSD', 'NEOTUSD', 'PAXTUSD', 'USDCTUSD', 'USDCPAX', 'LINKUSDT', 'LINKTUSD', 'LINKUSDC',
           'WAVESUSDT', 'WAVESTUSD', 'WAVESUSDC', 'BCHABCTUSD', 'BCHABCUSDC', 'BCHSVTUSD', 'BCHSVUSDC', 'LTCTUSD',
           'LTCUSDC', 'TRXUSDC', 'BTTUSDT', 'BNBUSDS', 'USDSUSDT', 'USDSPAX', 'USDSTUSD', 'USDSUSDC', 'BTTTUSD',
           'BTTUSDC', 'ONGUSDT', 'HOTUSDT', 'ZILUSDT', 'ZRXUSDT', 'FETUSDT', 'BATUSDT', 'XMRUSDT', 'ZECUSDT', 'ZECTUSD',
           'ZECUSDC', 'IOSTUSDT', 'CELRUSDT', 'ADAUSDC', 'NEOUSDC', 'DASHUSDT', 'NANOUSDT', 'OMGUSDT', 'THETAUSDT',
           'ENJUSDT', 'MITHUSDT', 'MATICUSDT', 'ATOMUSDT', 'ATOMUSDC', 'ATOMTUSD', 'ETCUSDC', 'ETCTUSD', 'BATUSDC',
           'BATTUSD', 'PHBUSDC', 'PHBTUSD', 'TFUELUSDT', 'TFUELUSDC', 'TFUELTUSD', 'ONEUSDT', 'ONETUSD', 'ONEUSDC',
           'FTMUSDT', 'FTMTUSD', 'FTMUSDC', 'BCPTTUSD', 'BCPTUSDC', 'ALGOUSDT', 'ALGOTUSD', 'ALGOUSDC', 'USDSBUSDT',
           'USDSBUSDS', 'GTOUSDT', 'GTOTUSD', 'GTOUSDC', 'ERDUSDT', 'ERDUSDC', 'DOGEUSDT', 'DOGEUSDC', 'DUSKUSDT',
           'DUSKUSDC', 'BGBPUSDC', 'ANKRUSDT', 'ANKRTUSD', 'ANKRUSDC', 'ONTUSDC', 'WINUSDT', 'WINUSDC', 'COSUSDT',
           'TUSDBTUSD', 'NPXSUSDT', 'NPXSUSDC', 'COCOSUSDT', 'MTLUSDT', 'TOMOUSDT', 'TOMOUSDC', 'PERLUSDC', 'PERLUSDT',
           'DENTUSDT', 'MFTUSDT', 'KEYUSDT', 'STORMUSDT', 'DOCKUSDT', 'WANUSDT', 'FUNUSDT', 'CVCUSDT', 'CHZUSDT',
           'BANDUSDT', 'BNBBUSD', 'BUSDUSDT', 'BEAMUSDT', 'XTZUSDT', 'RENUSDT', 'RVNUSDT', 'HCUSDT', 'HBARUSDT',
           'NKNUSDT', 'XRPBUSD', 'ETHBUSD', 'BCHABCBUSD', 'LTCBUSD', 'LINKBUSD', 'ETCBUSD', 'STXUSDT', 'KAVAUSDT',
           'BUSDNGN', 'ARPAUSDT', 'TRXBUSD','EOSBUSD', 'IOTXUSDT', 'RLCUSDT', 'MCOUSDT', 'XLMBUSD', 'ADABUSD',
          'CTXCUSDT', 'BCHUSDT', 'BCHUSDC', 'BCHTUSD', 'BCHBUSD', 'TROYUSDT', 'BUSDRUB', 'QTUMBUSD',
 'VETBUSD',
           'VITEUSDT', 'FTTUSDT', 'BUSDTRY', 'USDTTRY', 'USDTRUB', 'EURBUSD', 'EURUSDT', 'OGNUSDT', 'DREPUSDT',
           'BULLUSDT', 'BULLBUSD', 'BEARUSDT', 'BEARBUSD', 'ETHBULLUSDT', 'ETHBULLBUSD', 'ETHBEARUSDT', 'ETHBEARBUSD',
           'TCTUSDT', 'WRXUSDT', 'ICXBUSD', 'BTSUSDT', 'BTSBUSD', 'LSKUSDT', 'BNTUSDT', 'BNTBUSD', 'LTOUSDT',
           'ATOMBUSD', 'DASHBUSD', 'NEOBUSD', 'WAVESBUSD', 'XTZBUSD', 'EOSBULLUSDT', 'EOSBULLBUSD', 'EOSBEARUSDT',
           'EOSBEARBUSD', 'XRPBULLUSDT', 'XRPBULLBUSD', 'XRPBEARUSDT', 'XRPBEARBUSD', 'BATBUSD', 'ENJBUSD', 'NANOBUSD',
             'ONTBUSD', 'RVNBUSD', 'STRATBUSD', 'STRATUSDT', 'AIONBUSD', 'AIONUSDT', 'MBLUSDT', 'COTIUSDT', 'ALGOBUSD',
           'BTTBUSD', 'TOMOBUSD', 'XMRBUSD', 'ZECBUSD', 'BNBBULLUSDT', 'BNBBULLBUSD', 'BNBBEARUSDT', 'BNBBEARBUSD',
           'STPTUSDT', 'USDTZAR', 'BUSDZAR', 'WTCUSDT', 'DATABUSD', 'DATAUSDT', 'XZCUSDT', 'SOLUSDT', 'SOLBUSD',
           'USDTIDRT', 'BUSDIDRT', 'CTSIUSDT', 'CTSIBUSD', 'HIVEUSDT', 'CHRUSDT', 'GXSUSDT', 'ARDRUSDT', 'ERDBUSD',
           'LENDUSDT', 'HBARBUSD', 'MATICBUSD', 'WRXBUSD', 'ZILBUSD', 'MDTUSDT', 'STMXUSDT', 'KNCBUSD', 'KNCUSDT',
           'REPBUSD', 'REPUSDT', 'LRCBUSD', 'LRCUSDT', 'IQBUSD', 'PNTUSDT', 'GBPBUSD', 'DGBBUSD', 'USDTUAH', 'COMPBUSD',
           'COMPUSDT', 'BUSDBIDR', 'USDTBIDR', 'BKRWUSDT', 'BKRWBUSD', 'SCUSDT', 'ZENUSDT', 'SXPBUSD', 'SNXBUSD',
           'SNXUSDT', 'ETHUPUSDT', 'ETHDOWNUSDT', 'ADAUPUSDT', 'ADADOWNUSDT', 'LINKUPUSDT', 'LINKDOWNUSDT', 'VTHOBUSD',
           'VTHOUSDT', 'DCRBUSD', 'DGBUSDT', 'GBPUSDT', 'STORJBUSD', 'SXPUSDT', 'IRISBUSD', 'MKRUSDT', 'MKRBUSD',
           'DAIUSDT', 'DAIBUSD', 'RUNEBUSD', 'MANABUSD', 'DOGEBUSD', 'LENDBUSD', 'ZRXBUSD', 'DCRUSDT', 'STORJUSDT',
           'AUDBUSD', 'FIOBUSD', 'BNBUPUSDT', 'BNBDOWNUSDT', 'XTZUPUSDT', 'XTZDOWNUSDT', 'AVABUSD', 'USDTBKRW',
          'BUSDBKRW', 'IOTABUSD', 'MANAUSDT', 'AUDUSDT', 'BALBUSD', 'YFIBUSD', 'YFIUSDT', 'BLZBUSD', 'KMDBUSD',
          'BALUSDT', 'BLZUSDT', 'IRISUSDT', 'KMDUSDT', 'USDTDAI', 'BUSDDAI', 'JSTBUSD', 'JSTUSDT', 'SRMBUSD',
          'SRMUSDT', 'ANTBUSD', 'ANTUSDT', 'CRVBUSD', 'CRVUSDT', 'SANDUSDT', 'SANDBUSD', 'OCEANBUSD', 'OCEANUSDT',
          'NMRBUSD', 'NMRUSDT', 'DOTBUSD', 'DOTUSDT', 'LUNABUSD', 'LUNAUSDT', 'IDEXBUSD', 'RSRBUSD', 'RSRUSDT',
          'PAXGBUSD', 'PAXGUSDT', 'WNXMBUSD', 'WNXMUSDT', 'TRBBUSD', 'TRBUSDT', 'BZRXBUSD', 'BZRXUSDT', 'SUSHIBUSD',
          'SUSHIUSDT', 'YFIIBUSD', 'YFIIUSDT', 'KSMBUSD', 'KSMUSDT', 'EGLDBUSD', 'EGLDUSDT', 'DIABUSD', 'DIAUSDT',
          'RUNEUSDT', 'FIOUSDT', 'UMAUSDT', 'EOSUPUSDT', 'EOSDOWNUSDT', 'TRXUPUSDT', 'TRXDOWNUSDT', 'XRPUPUSDT',
          'XRPDOWNUSDT', 'DOTUPUSDT', 'DOTDOWNUSDT', 'USDTNGN', 'BELBUSD',
           'BELUSDT', 'SWRVBUSD', 'WINGBUSD', 'WINGUSDT', 'LTCUPUSDT', 'LTCDOWNUSDT', 'CREAMBUSD', 'UNIBUSD', 'UNIUSDT', 'NBSUSDT', 'OXTUSDT', 'SUNUSDT',
           'AVAXBUSD', 'AVAXUSDT', 'HNTUSDT', 'FLMBUSD', 'FLMUSDT', 'CAKEBUSD', 'UNIUPUSDT', 'UNIDOWNUSDT', 'ORNUSDT',
           'UTKUSDT', 'XVSBUSD', 'XVSUSDT', 'ALPHABUSD', 'ALPHAUSDT', 'VIDTBUSD', 'USDTBRL', 'AAVEBUSD', 'AAVEUSDT',
           'NEARBUSD', 'NEARUSDT', 'SXPUPUSDT', 'SXPDOWNUSDT', 'FILBUSD', 'FILUSDT', 'FILUPUSDT', 'FILDOWNUSDT',
           'YFIUPUSDT', 'YFIDOWNUSDT', 'INJBUSD', 'INJUSDT', 'AERGOBUSD', 'ONEBUSD', 'AUDIOBUSD', 'AUDIOUSDT',
           'CTKBUSD', 'CTKUSDT', 'BCHUPUSDT', 'BCHDOWNUSDT', 'BOTBUSD', 'AKROUSDT', 'KP3RBUSD', 'AXSBUSD', 'AXSUSDT',
           'HARDBUSD', 'HARDUSDT', 'DNTBUSD', 'DNTUSDT', 'CVPBUSD', 'STRAXBUSD', 'STRAXUSDT', 'FORBUSD', 'UNFIBUSD',
           'UNFIUSDT', 'FRONTBUSD', 'BCHABUSD', 'ROSEBUSD', 'ROSEUSDT', 'BUSDBRL', 'AVAUSDT', 'SYSBUSD', 'XEMUSDT',
           'HEGICBUSD', 'AAVEUPUSDT', 'AAVEDOWNUSDT', 'PROMBUSD', 'SKLBUSD', 'SKLUSDT', 'SUSDETH', 'SUSDUSDT',
           'COVERBUSD', 'GHSTBUSD', 'SUSHIUPUSDT', 'SUSHIDOWNUSDT', 'XLMUPUSDT', 'XLMDOWNUSDT', 'DFBUSD', 'GRTUSDT',
           'JUVBUSD', 'JUVUSDT', 'PSGBUSD', 'PSGUSDT', 'BUSDBVND', 'USDTBVND', '1INCHUSDT', 'REEFUSDT', 'OGUSDT',
           'ATMUSDT', 'ASRUSDT', 'CELOUSDT', 'RIFUSDT', 'TRUBUSD', 'TRUUSDT', 'DEXEBUSD', 'USDCBUSD', 'TUSDBUSD',
           'PAXBUSD', 'CKBBUSD', 'CKBUSDT', 'TWTBUSD', 'TWTUSDT', 'FIROUSDT', 'LITBUSD', 'LITUSDT', 'BUSDVAI',
           'SFPBUSD', 'SFPUSDT', 'FXSBUSD', 'DODOBUSD', 'DODOUSDT', 'CAKEUSDT', 'BAKEBUSD', 'UFTBUSD', '1INCHBUSD',
           'BANDBUSD', 'GRTBUSD', 'IOSTBUSD', 'OMGBUSD', 'REEFBUSD', 'ACMBUSD', 'ACMUSDT', 'AUCTIONBUSD', 'PHABUSD',
           'TVKBUSD', 'BADGERBUSD', 'BADGERUSDT', 'FISBUSD', 'FISUSDT', 'OMBUSD', 'OMUSDT', 'PONDBUSD', 'PONDUSDT',
           'DEGOBUSD', 'DEGOUSDT', 'ALICEBUSD', 'ALICEUSDT', 'CHZBUSD', 'BIFIBUSD', 'LINABUSD', 'LINAUSDT', 'PERPBUSD',
           'PERPUSDT', 'RAMPBUSD', 'RAMPUSDT', 'SUPERBUSD', 'SUPERUSDT', 'CFXBUSD', 'CFXUSDT', 'XVGBUSD', 'EPSBUSD',
           'EPSUSDT', 'AUTOBUSD', 'AUTOUSDT', 'TKOBUSD', 'TKOUSDT', 'PUNDIXUSDT', 'TLMBUSD', 'TLMUSDT', '1INCHUPUSDT',
           '1INCHDOWNUSDT', 'BTGBUSD', 'BTGUSDT', 'HOTBUSD', 'MIRBUSD', 'MIRUSDT', 'BARBUSD', 'BARUSDT', 'FORTHBUSD',
           'FORTHUSDT', 'BAKEUSDT', 'BURGERBUSD', 'BURGERUSDT', 'SLPBUSD', 'SLPUSDT', 'SHIBUSDT', 'SHIBBUSD', 'ICPBUSD',
           'ICPUSDT', 'USDTGYEN', 'ARBUSD', 'ARUSDT']
# # BTC
# symbols = ['ETHBTC', 'LTCBTC', 'BNBBTC', 'NEOBTC', 'BCCBTC', 'GASBTC', 'BTCUSDT', 'HSRBTC', 'MCOBTC', 'WTCBTC',
#            'LRCBTC', 'QTUMBTC', 'YOYOBTC', 'OMGBTC', 'ZRXBTC', 'STRATBTC', 'SNGLSBTC', 'BQXBTC', 'KNCBTC', 'FUNBTC',
#            'SNMBTC', 'IOTABTC', 'LINKBTC', 'XVGBTC', 'SALTBTC', 'MDABTC', 'MTLBTC', 'SUBBTC', 'EOSBTC', 'SNTBTC',
#            'ETCBTC', 'MTHBTC', 'ENGBTC', 'DNTBTC', 'ZECBTC', 'BNTBTC', 'ASTBTC', 'DASHBTC', 'OAXBTC', 'ICNBTC',
#            'BTGBTC', 'EVXBTC', 'REQBTC', 'VIBBTC', 'TRXBTC',
# symbols = ['POWRBTC', 'ARKBTC', 'XRPBTC', 'MODBTC', 'ENJBTC',
#            'STORJBTC', 'VENBTC', 'KMDBTC', 'RCNBTC', 'NULSBTC', 'RDNBTC', 'XMRBTC', 'DLTBTC', 'AMBBTC', 'BATBTC',
#            'BCPTBTC', 'ARNBTC', 'GVTBTC', 'CDTBTC', 'GXSBTC', 'POEBTC', 'QSPBTC', 'BTSBTC', 'XZCBTC', 'LSKBTC',
#            'TNTBTC', 'FUELBTC', 'MANABTC', 'BCDBTC', 'DGDBTC', 'ADXBTC', 'ADABTC', 'PPTBTC', 'CMTBTC', 'XLMBTC',
#            'CNDBTC', 'LENDBTC', 'WABIBTC', 'TNBBTC', 'WAVESBTC', 'GTOBTC', 'ICXBTC', 'OSTBTC', 'ELFBTC', 'AIONBTC',
#            'NEBLBTC', 'BRDBTC', 'EDOBTC', 'WINGSBTC', 'NAVBTC', 'LUNBTC', 'TRIGBTC', 'APPCBTC', 'VIBEBTC', 'RLCBTC',
#            'INSBTC', 'PIVXBTC', 'IOSTBTC', 'CHATBTC', 'STEEMBTC', 'NANOBTC', 'VIABTC', 'BLZBTC', 'AEBTC', 'RPXBTC',
#            'NCASHBTC', 'POABTC', 'ZILBTC', 'ONTBTC', 'STORMBTC', 'XEMBTC', 'WANBTC', 'WPRBTC', 'QLCBTC', 'SYSBTC',
#            'GRSBTC', 'CLOAKBTC', 'GNTBTC', 'LOOMBTC', 'BCNBTC', 'REPBTC', 'BTCTUSD', 'TUSDBTC', 'ZENBTC', 'SKYBTC',
#            'CVCBTC', 'THETABTC', 'IOTXBTC', 'QKCBTC', 'AGIBTC', 'NXSBTC', 'DATABTC', 'SCBTC', 'NPXSBTC', 'KEYBTC',
#            'NASBTC', 'MFTBTC', 'DENTBTC', 'ARDRBTC', 'HOTBTC', 'VETBTC', 'DOCKBTC', 'POLYBTC', 'PHXBTC', 'HCBTC',
#            'GOBTC', 'PAXBTC', 'RVNBTC', 'DCRBTC', 'MITHBTC', 'BCHABCBTC', 'BCHSVBTC', 'BTCPAX', 'RENBTC', 'BTCUSDC',
#            'BTTBTC', 'BTCUSDS', 'ONGBTC', 'FETBTC', 'CELRBTC', 'MATICBTC', 'ATOMBTC', 'PHBBTC', 'TFUELBTC', 'ONEBTC',
#            'FTMBTC', 'BTCBBTC', 'ALGOBTC', 'ERDBTC', 'DOGEBTC', 'DUSKBTC', 'ANKRBTC', 'WINBTC', 'COSBTC', 'COCOSBTC',
#            'TOMOBTC', 'PERLBTC', 'CHZBTC', 'BANDBTC', 'BTCBUSD', 'BEAMBTC', 'XTZBTC', 'HBARBTC', 'NKNBTC', 'STXBTC',
#            'KAVABTC', 'BTCNGN', 'ARPABTC', 'CTXCBTC', 'BCHBTC', 'BTCRUB', 'TROYBTC', 'VITEBTC', 'FTTBTC', 'BTCTRY',
#            'BTCEUR', 'OGNBTC', 'DREPBTC', 'TCTBTC', 'WRXBTC', 'LTOBTC', 'MBLBTC', 'COTIBTC', 'STPTBTC', 'BTCZAR',
#            'BTCBKRW', 'SOLBTC', 'BTCIDRT', 'CTSIBTC', 'HIVEBTC', 'CHRBTC', 'BTCUPUSDT', 'BTCDOWNUSDT', 'MDTBTC',
#            'STMXBTC', 'PNTBTC', 'BTCGBP', 'DGBBTC', 'BTCUAH', 'COMPBTC', 'BTCBIDR', 'SXPBTC', 'SNXBTC', 'IRISBTC',
#            'MKRBTC', 'DAIBTC', 'RUNEBTC', 'BTCAUD', 'FIOBTC', 'AVABTC', 'BALBTC', 'YFIBTC', 'BTCDAI', 'JSTBTC',
#            'SRMBTC', 'ANTBTC', 'CRVBTC', 'SANDBTC', 'OCEANBTC', 'NMRBTC', 'DOTBTC', 'LUNABTC', 'IDEXBTC', 'RSRBTC',
#            'PAXGBTC', 'WNXMBTC', 'TRBBTC', 'BZRXBTC', 'WBTCBTC', 'WBTCETH', 'SUSHIBTC', 'YFIIBTC', 'KSMBTC', 'EGLDBTC',
#            'DIABTC', 'UMABTC', 'BELBTC', 'WINGBTC', 'UNIBTC', 'NBSBTC', 'OXTBTC', 'SUNBTC', 'AVAXBTC', 'HNTBTC',
#            'FLMBTC', 'SCRTBTC', 'ORNBTC', 'UTKBTC', 'XVSBTC', 'ALPHABTC', 'VIDTBTC', 'BTCBRL', 'AAVEBTC', 'NEARBTC',
#            'FILBTC', 'INJBTC', 'AERGOBTC', 'AUDIOBTC', 'CTKBTC', 'BOTBTC', 'AKROBTC', 'AXSBTC', 'HARDBTC', 'RENBTCBTC',
#            'RENBTCETH', 'STRAXBTC', 'FORBTC', 'UNFIBTC', 'ROSEBTC', 'SKLBTC', 'SUSDBTC', 'GLMBTC', 'GRTBTC', 'JUVBTC',
#            'PSGBTC', '1INCHBTC', 'REEFBTC', 'OGBTC', 'ATMBTC', 'ASRBTC', 'CELOBTC', 'RIFBTC', 'BTCSTBTC', 'BTCSTBUSD',
#            'BTCSTUSDT', 'TRUBTC', 'CKBBTC', 'TWTBTC', 'FIROBTC', 'LITBTC', 'BTCVAI', 'SFPBTC', 'FXSBTC', 'DODOBTC',
#            'FRONTBTC', 'EASYBTC', 'CAKEBTC', 'ACMBTC', 'AUCTIONBTC', 'PHABTC', 'TVKBTC', 'BADGERBTC', 'FISBTC', 'OMBTC',
#            'PONDBTC', 'DEGOBTC', 'ALICEBTC', 'LINABTC', 'PERPBTC', 'RAMPBTC', 'SUPERBTC', 'CFXBTC', 'EPSBTC', 'AUTOBTC',
#            'TKOBTC', 'TLMBTC', 'MIRBTC', 'BARBTC', 'FORTHBTC', 'EZBTC', 'ICPBTC', 'BTCGYEN', 'ARBTC',
#            'QTUMETH', 'EOSETH', 'SNTETH', 'BNTETH', 'BNBETH', 'OAXETH', 'DNTETH', 'MCOETH', 'ICNETH', 'WTCETH',
#            'LRCETH', 'OMGETH', 'ZRXETH', 'STRATETH', 'SNGLSETH', 'BQXETH', 'KNCETH', 'FUNETH', 'SNMETH', 'NEOETH',
#            'IOTAETH', 'LINKETH', 'XVGETH', 'SALTETH', 'MDAETH', 'MTLETH', 'SUBETH', 'ETCETH', 'MTHETH', 'ENGETH',
#            'ZECETH', 'ASTETH', 'DASHETH', 'BTGETH', 'EVXETH', 'REQETH', 'VIBETH', 'HSRETH', 'TRXETH', 'POWRETH',
#            'ARKETH', 'YOYOETH', 'XRPETH', 'MODETH', 'ENJETH', 'STORJETH', 'VENETH', 'KMDETH', 'RCNETH', 'NULSETH',
#            'RDNETH', 'XMRETH', 'DLTETH', 'AMBETH', 'BCCETH', 'BATETH', 'BCPTETH', 'ARNETH', 'GVTETH', 'CDTETH',
#            'GXSETH', 'POEETH', 'QSPETH', 'BTSETH', 'XZCETH', 'LSKETH', 'TNTETH', 'FUELETH', 'MANAETH', 'BCDETH',
#            'DGDETH', 'ADXETH', 'ADAETH', 'PPTETH', 'CMTETH', 'XLMETH', 'CNDETH', 'LENDETH', 'WABIETH', 'LTCETH',
#            'TNBETH', 'WAVESETH', 'GTOETH', 'ICXETH', 'OSTETH', 'ELFETH', 'AIONETH', 'NEBLETH', 'BRDETH', 'EDOETH',
#            'WINGSETH', 'NAVETH', 'LUNETH', 'TRIGETH', 'APPCETH', 'VIBEETH', 'RLCETH', 'INSETH', 'PIVXETH', 'IOSTETH',
#            'CHATETH', 'STEEMETH', 'NANOETH', 'VIAETH', 'BLZETH', 'AEETH', 'RPXETH', 'NCASHETH', 'POAETH', 'ZILETH',
#            'ONTETH', 'STORMETH', 'XEMETH', 'WANETH', 'WPRETH', 'QLCETH', 'SYSETH', 'GRSETH', 'CLOAKETH', 'GNTETH',
#            'LOOMETH', 'BCNETH', 'REPETH', 'ZENETH', 'SKYETH', 'CVCETH', 'THETAETH', 'IOTXETH', 'QKCETH', 'AGIETH',
#            'NXSETH', 'DATAETH', 'SCETH', 'NPXSETH', 'KEYETH', 'NASETH', 'MFTETH',
#            'DENTETH', 'ARDRETH', 'HOTETH',
#            'VETETH', 'DOCKETH', 'PHXETH', 'HCETH', 'PAXETH', 'ETHPAX', 'ETHRUB', 'ETHTRY', 'ETHEUR', 'ETHZAR',
#            'ETHBKRW', 'STMXETH', 'ETHGBP', 'ETHBIDR', 'ETHAUD', 'ETHDAI', 'ETHNGN', 'SCRTETH', 'AAVEETH', 'EASYETH',
#            'ETHBRL', 'SLPETH', 'CVPETH', 'STRAXETH', 'FRONTETH', 'HEGICETH', 'COVERETH', 'GLMETH', 'GHSTETH', 'DFETH',
#            'GRTETH', 'DEXEETH', 'FIROETH', 'BETHETH', 'PROSETH', 'UFTETH', 'PUNDIXETH', 'EZETH',
#            'AAVEBNB', 'ADABNB', 'ADXBNB', 'AEBNB', 'AGIBNB', 'AIONBNB', 'ALGOBNB', 'ALPHABNB', 'AMBBNB', 'ANKRBNB',
#            'ANTBNB', 'APPCBNB', 'ARBNB', 'ARDRBNB', 'ARPABNB', 'ATOMBNB', 'AVABNB', 'AVAXBNB', 'AXSBNB', 'BAKEBNB',
#            'BALBNB', 'BANDBNB', 'BATBNB', 'BCCBNB', 'BCHBNB', 'BCNBNB', 'BCPTBNB', 'BEAMBNB', 'BELBNB', 'BIFIBNB',
#            'BLZBNB', 'BNBAUD', 'BNBBIDR', 'BNBBKRW', 'BNBBRL', 'BNBDAI', 'BNBEUR', 'BNBGBP', 'BNBIDRT', 'BNBNGN',
#            'BNBPAX', 'BNBRUB', 'BNBTRY', 'BNBUAH', 'BNBZAR', 'BRDBNB', 'BTSBNB', 'BTTBNB', 'BURGERBNB', 'BZRXBNB',
#            'CAKEBNB', 'CELRBNB', 'CHRBNB', 'CHZBNB', 'CMTBNB', 'CNDBNB', 'COCOSBNB', 'COMPBNB', 'COSBNB', 'COTIBNB',
#            'CREAMBNB', 'CRVBNB', 'CTKBNB', 'CTSIBNB', 'CTXCBNB', 'CVCBNB', 'DAIBNB', 'DASHBNB', 'DCRBNB', 'DGBBNB',
#            'DIABNB', 'DLTBNB', 'DOGEBNB', 'DOTBNB', 'DREPBNB', 'DUSKBNB', 'EGLDBNB', 'ENJBNB', 'EOSBNB', 'ERDBNB',
#            'ETCBNB', 'FETBNB', 'FILBNB', 'FIOBNB', 'FLMBNB', 'FTMBNB', 'FTTBNB', 'GNTBNB', 'GOBNB', 'GTOBNB', 'HARDBNB',
#            'HBARBNB', 'HIVEBNB', 'HOTBNB', 'ICPBNB', 'ICXBNB', 'INJBNB', 'IOSTBNB', 'IOTABNB', 'IQBNB', 'IRISBNB',
#            'JSTBNB', 'KAVABNB', 'KP3RBNB', 'KSMBNB', 'LOOMBNB', 'LSKBNB', 'LTCBNB', 'LTOBNB', 'LUNABNB', 'MATICBNB',
#            'MBLBNB', 'MCOBNB', 'MDTBNB', 'MFTBNB', 'MITHBNB', 'MKRBNB', 'NANOBNB', 'NASBNB', 'NAVBNB', 'NCASHBNB',
#            'NEARBNB', 'NEBLBNB', 'NEOBNB', 'NKNBNB', 'NMRBNB', 'NULSBNB', 'NXSBNB', 'OCEANBNB', 'OGNBNB', 'OMGBNB',
#            'ONEBNB', 'ONGBNB', 'ONTBNB', 'OSTBNB', 'PAXBNB', 'PAXGBNB', 'PERLBNB', 'PHBBNB', 'PHXBNB', 'PIVXBNB',
#            'POABNB', 'POLYBNB', 'POWRBNB', 'PROMBNB', 'QLCBNB', 'QSPBNB', 'QTUMBNB', 'RCNBNB', 'RDNBNB', 'RENBNB',
#            'REPBNB', 'RLCBNB', 'RPXBNB', 'RSRBNB', 'RUNEBNB', 'RVNBNB', 'SANDBNB', 'SCBNB', 'SKYBNB', 'SNXBNB',
#            'SOLBNB', 'SPARTABNB', 'SRMBNB', 'STEEMBNB', 'STMXBNB', 'STORMBNB', 'STPTBNB', 'STRATBNB', 'STXBNB',
#            'SUSHIBNB', 'SWRVBNB', 'SXPBNB', 'SYSBNB', 'TCTBNB', 'TFUELBNB', 'THETABNB', 'TOMOBNB', 'TRBBNB', 'TRIGBNB',
#            'TROYBNB', 'TRXBNB', 'UNFIBNB', 'UNIBNB', 'VENBNB', 'VETBNB', 'VIABNB', 'VITEBNB', 'VTHOBNB', 'WABIBNB',
#            'WANBNB', 'WAVESBNB', 'WINBNB', 'WINGBNB', 'WNXMBNB', 'WRXBNB', 'WTCBNB', 'XEMBNB', 'XLMBNB', 'XMRBNB',
#            'XRPBNB', 'XTZBNB', 'XVSBNB', 'XZCBNB', 'YFIBNB', 'YFIIBNB', 'YOYOBNB', 'ZECBNB', 'ZENBNB', 'ZILBNB',
#            'ZRXBNB','AAVEBKRW', 'ADAAUD', 'ADABKRW', 'ADABRL', 'ADAEUR', 'ADAGBP', 'ADAPAX', 'ADARUB', 'ADATRY', 'ALGOPAX',
#            'ANKRPAX', 'ATOMPAX', 'AVAXEUR', 'AVAXTRY', 'BATPAX', 'BCHABCPAX', 'BCHEUR', 'BCHPAX', 'BCHSVPAX', 'BCPTPAX',
#            'BTTBRL', 'BTTEUR', 'BTTPAX', 'BTTTRX', 'BTTTRY', 'CAKEGBP', 'CHZBRL', 'CHZEUR', 'CHZGBP', 'CHZTRY',
#            'DOGEAUD', 'DOGEBIDR', 'DOGEBRL', 'DOGEEUR', 'DOGEGBP', 'DOGEPAX', 'DOGERUB', 'DOGETRY', 'DOTBIDR',
#            'DOTBKRW', 'DOTBRL', 'DOTEUR', 'DOTGBP', 'DOTNGN', 'DOTTRY', 'DUSKPAX', 'EGLDEUR', 'ENJBRL', 'ENJEUR',
#            'ENJGBP', 'EOSEUR', 'EOSPAX', 'EOSTRY', 'ERDPAX', 'ETCBRL', 'ETCEUR', 'ETCPAX', 'FTMPAX', 'GRTEUR', 'GTOPAX',
#            'HOTBRL', 'HOTEUR', 'HOTTRY', 'LENDBKRW', 'LINKAUD', 'LINKBKRW', 'LINKBRL', 'LINKEUR', 'LINKGBP', 'LINKNGN',
#            'LINKPAX', 'LINKTRY', 'LTCBRL', 'LTCEUR', 'LTCGBP', 'LTCNGN', 'LTCPAX', 'LTCRUB', 'LUNAEUR', 'MATICEUR',
#            'NEOPAX', 'NEOTRY', 'ONEBIDR', 'ONEPAX', 'ONTPAX', 'ONTTRY', 'PHBPAX', 'RVNTRY', 'SHIBEUR', 'SHIBRUB',
#            'SRMBIDR', 'SXPAUD', 'SXPBIDR', 'SXPEUR', 'SXPGBP', 'SXPTRY', 'TFUELPAX', 'THETAEUR', 'TKOBIDR', 'TRXAUD',
#            'TRXEUR', 'TRXNGN', 'TRXPAX', 'TRXTRY', 'TRXXRP', 'UNIEUR', 'VETEUR', 'VETGBP', 'VETTRY', 'WAVESPAX',
#            'WINBRL', 'WINEUR', 'WINTRX', 'WRXEUR', 'XLMEUR', 'XLMPAX', 'XLMTRY', 'XRPAUD', 'XRPBKRW', 'XRPBRL',
#            'XRPEUR', 'XRPGBP', 'XRPNGN', 'XRPPAX', 'XRPRUB', 'XRPTRY', 'XZCXRP', 'YFIEUR', 'ZECPAX', 'ZILBIDR']