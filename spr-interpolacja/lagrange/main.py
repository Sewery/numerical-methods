import lagrangeCzebyszew as lc
import lagrangeRownomiernie as lr
import newtonCzebyszew as nc
import newtonRownomiernie as nr
import tabelaGenerowanieCzebyszew as tgc
import tabelaGenerowanieRownomiernie as tgr

tgc.export_error_table_to_csv([i for i in range(2,36,1)],"./data/chebysev-out-2-35-1")
tgc.export_error_table_to_csv([i for i in range(40,61,5)],"./data/chebysev-out-40-60-5")
tgc.export_error_table_to_csv([i for i in range(60,210,10)],"./data/evenly-out-60-200-10")


tgr.export_error_table_to_csv([i for i in range(2,36,1)],"./data/chebysev-out-2-35-1")
tgr.export_error_table_to_csv([i for i in range(40,61,5)],"./data/chebysev-out-40-60-5")
tgr.export_error_table_to_csv([i for i in range(60,210,10)],"./data/evenly-out-60-200-10")


lr.print_plots([2,4,10])
lr.print_plots([5,8])
lr.print_plots([11,15])
lr.print_plots([20,40,65,75])

nr.print_plots([2,4,10])
nr.print_plots([5,8])
nr.print_plots([11,15])
nr.print_plots([20,40,65,75])

lc.print_plots([5,8])
lc.print_plots([11,15,20])
lc.print_plots([40,100])

nc.print_plots([5,8])
nc.print_plots([11,15,20])
nc.print_plots([40,100])