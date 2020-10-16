puts "applying partitioning constraints"

# create_pblock pblock_X0_Y0
# resize_pblock pblock_X0_Y0 -add CLOCKREGION_X0Y0:CLOCKREGION_X3Y3
#
# create_pblock pblock_X0_Y1
# resize_pblock pblock_X0_Y1 -add CLOCKREGION_X0Y4:CLOCKREGION_X3Y7
#
# create_pblock pblock_X0_Y2
# resize_pblock pblock_X0_Y2 -add CLOCKREGION_X0Y8:CLOCKREGION_X3Y11

create_pblock pblock_X1_Y0
resize_pblock pblock_X1_Y0 -add { \
  SLICE_X117Y0:SLICE_X205Y239 \
  DSP48E2_X16Y0:DSP48E2_X29Y89 \
  LAGUNA_X16Y0:LAGUNA_X27Y119 \
  RAMB18_X8Y0:RAMB18_X11Y95 \
  RAMB36_X8Y0:RAMB36_X11Y47 \
  URAM288_X2Y0:URAM288_X4Y63 \
  SLICE_X206Y0:SLICE_X232Y59 \
  DSP48E2_X30Y0:DSP48E2_X31Y17 \
  PCIE4CE4_X1Y0:PCIE4CE4_X1Y0 \
  RAMB18_X12Y0:RAMB18_X13Y23 \
  RAMB36_X12Y0:RAMB36_X13Y11 \
}

# create_pblock pblock_X1_Y1
# resize_pblock pblock_X1_Y1 -add { \
#   SLICE_X117Y240:SLICE_X205Y479 \
#   DSP48E2_X16Y90:DSP48E2_X29Y185 \
#   LAGUNA_X16Y120:LAGUNA_X27Y359 \
#   RAMB18_X8Y96:RAMB18_X11Y191 \
#   RAMB36_X8Y48:RAMB36_X11Y95 \
#   URAM288_X2Y64:URAM288_X4Y127 \
# }
#
# create_pblock pblock_X1_Y2
# resize_pblock pblock_X1_Y2 -add { \
#   SLICE_X117Y480:SLICE_X205Y719 \
#   DSP48E2_X16Y186:DSP48E2_X29Y281 \
#   LAGUNA_X16Y360:LAGUNA_X27Y599 \
#   RAMB18_X8Y192:RAMB18_X11Y287 \
#   RAMB36_X8Y96:RAMB36_X11Y143 \
#   URAM288_X2Y128:URAM288_X4Y191 \
# }

add_cells_to_pblock [get_pblocks pblock_X1_Y0] [get_cells -regex {
  pfm_top_i/dynamic_region/SSSP
}]
