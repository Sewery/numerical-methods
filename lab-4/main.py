import hermiteCzebyszew as hc
import hermiteRownomiernie as hr
import tabelaGenerowanieCzebyszew as tgc
import tabelaGenerowanieRownomiernie as tgr

tgc.export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-co-drugi",2)
tgc.export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-kazdy",1)
tgc.export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-co-drugi",2)
tgc.export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-kazdy",1)
tgc.export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-co-drugi",2)
tgc.export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-kazdy",1)


tgr.export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-co-drugi",2)
tgr.export_error_table_to_csv([i for i in range(2,36,1)],"./data/czeb-hermite-out-2-35-1-kazdy",1)
tgr.export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-co-drugi",2)
tgr.export_error_table_to_csv([i for i in range(40,61,5)],"./data/czeb-hermite-out-40-60-5-kazdy",1)
tgr.export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-co-drugi",2)
tgr.export_error_table_to_csv([i for i in range(60,180,10)],"./data/czeb-hermite-out-60-180-10-kazdy",1)


hc.print_plots([2,5,9])
hc.print_plots([11,15])
hc.print_plots([20,40,80])

hr.print_plots([2,5,9])
hr.print_plots([11,15])
hr.print_plots([20,40,80])
