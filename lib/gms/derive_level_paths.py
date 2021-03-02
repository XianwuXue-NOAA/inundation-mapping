#!/usr/bin/env python3

from stream_branches import StreamNetwork
import argparse


def Derive_level_paths(in_stream_network, out_stream_network,
                       out_stream_network_dissolved=None,
                       toNode_attribute='To_Node',fromNode_attribute='From_Node',
                       verbose=False
                       ):
    # load file
    stream_network = StreamNetwork.from_file(in_stream_network)

    # derive outlets and inlets
    stream_network = stream_network.derive_outlets(toNode_attribute,fromNode_attribute,verbose=verbose)
    stream_network = stream_network.derive_inlets(toNode_attribute,fromNode_attribute,verbose=verbose)

    # derive arbolate sum
    stream_network = stream_network.get_arbolate_sum(arbolate_sum_attribute='arbolate_sum',
                                                     inlets_attribute='inlet_id',
                                                     reach_id_attribute='HydroID',
                                                     toNode_attribute=toNode_attribute,
                                                     fromNode_attribute=fromNode_attribute,
                                                     length_conversion_factor_to_km = 0.001,
                                                     verbose=verbose
                                                    )

    # derive stream branches
    stream_network = stream_network.derive_stream_branches( toNode_attribute=toNode_attribute,
                                                            fromNode_attribute=fromNode_attribute,
                                                            branch_id_attribute='levpa_id',
                                                            reach_id_attribute='HydroID',
                                                            comparison_attributes='arbolate_sum',
                                                            comparison_function=max,
                                                            verbose=verbose
                                                           )
    
    if out_stream_network is not None:
        stream_network.write(out_stream_network)
    
    if out_stream_network_dissolved is not None:
    
        # dissolve by levelpath
        stream_network = stream_network.dissolve_by_branch(branch_id_attribute='levpa_id',
                                                           attribute_excluded=None, #'order_',
                                                           values_excluded=None, #[1,2],
                                                           out_vector_files=out_stream_network_dissolved,
                                                           verbose=verbose)

    return(stream_network)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create stream network level paths')
    parser.add_argument('-i','--in-stream-network', help='Input stream network', required=True)
    parser.add_argument('-o','--out-stream-network', help='Output stream network', required=True)
    parser.add_argument('-d','--out-stream-network-dissolved', help='Dissolved output stream network', required=False,default=None)
    parser.add_argument('-v','--verbose', help='Verbose output', required=False,default=False,action='store_true')
    
    args = vars(parser.parse_args())

    Derive_level_paths(**args)

