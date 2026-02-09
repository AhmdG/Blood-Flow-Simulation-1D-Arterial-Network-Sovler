
from PyPDF2 import PdfMerger

merger = PdfMerger()
for pdf in ["page_1_1.pdf", "page_2_2.pdf", "page_3_3.pdf", "page_4_4.pdf", "page_5_5.pdf"]:
    merger.append(pdf)

merger.write("merged.pdf")
merger.close()

from PyPDF2 import PdfReader, PdfWriter

input_pdf = "/home/ahmed/Downloads/kontoasuzug.pdf"  # Your PDF file
reader = PdfReader(input_pdf)

for i, page in enumerate(reader.pages, start=1):
    writer = PdfWriter()
    writer.add_page(page)
    
    output_filename = f"page_{i}.pdf"
    with open(output_filename, "wb") as output_pdf:
        writer.write(output_pdf)

print("Done! One PDF per page created.")
import torch 

def extract_aligned_conn_data(data, n_arteries=1):
    """
    Extracts and organizes connection data from a tensor, optimized for performance and memory.
    # [x/L, tt/T, artery_index/n_arteries, conn_0/n_arteries, conn_1/n_arteries, L, Rp, Rd, A0, dA0dx, dTaudx]
    
    Args:
        data: torch.Tensor of shape (N, 5+) with columns [x, t, portion, conn_0, conn_1, ...]
    
    Returns:
        
    """
    # Early validation
    if data.shape[1] != 12:
        raise ValueError("Input data must have at least 5 columns")
    
    # Extract columns with clear names
    x_values = data[:, 0]
    time_values = data[:, 1]
    portion_idx = (data[:, 2]*n_arteries).long()  # Portion index scaled to number of arteries
    conn_0 = (data[:, 3]*n_arteries).long()
    conn_1 = (data[:, 4]*n_arteries).long()

    # Filter edge rows (x == 1)
    edge_mask = x_values == 1
    #edge_data = data[edge_mask]
    edge_time = time_values[edge_mask]
    edge_conn_0 = conn_0[edge_mask]
    edge_conn_1 = conn_1[edge_mask]

    is_bifur = (edge_conn_0 != -1) & (edge_conn_1 != -1)

    # Extract connection indices and values
    bif_conn1_val = edge_conn_1[is_bifur]

        # Filter x == 0 data for connection lookups
    x0_mask = x_values == 0
    x0_idx = torch.where(x0_mask)[0]
    x0_data = data[x0_mask]
    x0_time = time_values[x0_mask]
    x0_portion = portion_idx[x0_mask]

    # Vectorized connection row lookup
    def find_conn_rows(target_times, target_portions):
        """
        Vectorized lookup for rows in x0_data matching given times and portion indices.
        
        Args:
            target_times: torch.Tensor of time values
            target_portions: torch.Tensor of portion indices
            
        Returns:
            torch.Tensor of matched rows or NaN-filled rows if no match
        """
        if target_times.numel() == 0:
            return torch.empty(0, data.shape[1], device=data.device, dtype=data.dtype)
        
        # Create broadcastable masks
        time_match = x0_time.view(-1, 1) == target_times
        portion_match = x0_portion.view(-1, 1) == target_portions
        matches = time_match & portion_match
        
        # Find first match per target
        match_idx = matches.any(dim=0).nonzero(as_tuple=True)[0]
        #result = torch.full((target_times.size(0), data.shape[1]), float('nan'), 
                           #device=data.device, dtype=data.dtype)
        result_idx = torch.full((target_times.size(0),), -1, 
                              dtype=torch.long, device=data.device)       
        if match_idx.numel() > 0:
            valid_matches = matches[:, match_idx]
            matched_rows = torch.where(valid_matches)[0]
            #result[match_idx] = x0_data[matched_rows]
            try:
                result_idx[match_idx] = x0_idx[matched_rows]
            except:
                pass
        
        #return result, result_idx
        return result_idx

    # Compute aligned connection rows
    bif_conn2_idx = find_conn_rows(edge_time[is_bifur], bif_conn1_val)
    return   bif_conn2_idx

inputs = torch.load('tensor.pt')
extract_aligned_conn_data(data=inputs, n_arteries=77)
for idx, i in enumerate(inputs):
    print(idx)
    extract_aligned_conn_data(data=i.unsqueeze(0), n_arteries=77)
    pass