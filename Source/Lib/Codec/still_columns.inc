/* AV2 v1 uniform columns divide full 64-pixel SBs evenly, then assign one
 * extra SB to the first columns and the partial right edge to the last. */
typedef struct StillColumn {
    SvtAv2Handle handle;
    SvtAv2ColumnPayload payload;
    uint32_t x;
    uint16_t* qindex;
    uint8_t* gdf_flags;
    SvtAv2Error error;
} StillColumn;

static SvtAv2Error encode_still_column(SvtAv2EncoderContext* parent, StillColumn* column,
    uint32_t width, uint32_t qindex) {
    SvtAv2Error err=svt_av2_enc_init_handle(&column->handle,NULL);
    if (err) return err;
    SvtAv2EncoderContext* child=(SvtAv2EncoderContext*)column->handle;
    child->config=parent->config;
    child->config.output_path=NULL;
    err=svt_av2_enc_set_parameter(child,SVT_AV2_PARAM_OUTPUT_PATH,&parent->config.output_path);
    if (err) return err;
    child->config.width=width;
    child->config.threads=1;
    child->full_picture_width=parent->frame_width;
    child->content_type=parent->content_type;
    child->rd_lambda_scale=parent->rd_lambda_scale;
    child->satd_angles=parent->satd_angles;
    err=svt_av2_enc_init(child);
    if (err) return err;
    /* The parent applies source filters before the frame is divided into tiles. */
    child->luma_compensation_scope=0;
    child->source_cdef_scope=0;
    SvtAv2Picture pic={0};
    pic.width=width;pic.height=parent->frame_height;
    pic.bit_depth=parent->config.bit_depth;pic.chroma_format=parent->config.chroma_format;
    const uint32_t step=pic.bit_depth==10u?2u:1u;
    for (unsigned p=0;p<3;++p) {
        pic.strides[p]=parent->frame_source.strides[p];
        pic.planes[p]=parent->frame_source.planes[p]+(size_t)(column->x>>(p!=0))*step;
    }
    err=svt_av2_enc_send_picture(child,&pic);
    if (err) return err;
    if (parent->still_aq) {
        const size_t count=(size_t)((width+63u)/64u)*((pic.height+63u)/64u);
        column->qindex=(uint16_t*)calloc(count,sizeof(*column->qindex));
        if (!column->qindex) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        child->column_qindex=column->qindex;
    }
    const SvtAv2PictureReference refs={0};
    err=encode_reconstruction(child,width,pic.height,qindex,&refs);
    if (err) return err;
    for (unsigned p=0;p<3;++p) {
        const unsigned shift=p!=0;
        const size_t bytes=(width>>shift)*step;
        for (uint32_t y=0;y<(pic.height>>shift);++y)
            memcpy(parent->recon.planes[p]+(size_t)y*parent->recon.strides[p]+(column->x>>shift)*step,
                child->recon.planes[p]+(size_t)y*child->recon.strides[p],bytes);
    }
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error pack_still_column(StillColumn* column) {
    SvtAv2EncoderContext* child=(SvtAv2EncoderContext*)column->handle;
    const SvtAv2FrameHeaderConfig* h=&child->still_tools_header;
    /* The entropy writer shares its implementation with a single-tile frame.
     * Strip its private header at the measured bit boundary. */
    uint8_t header[128];
    SvtAv2BitWriter bw;
    svt_av2_bit_writer_init(&bw,header,sizeof(header));
    if (svt_av2_bit_writer_write_bit(&bw,1u) || svt_av2_frame_header_write(&bw,h) ||
        svt_av2_bit_writer_byte_alignment(&bw)) return SVT_AV2_ERROR_INTERNAL;
    const size_t skip=svt_av2_bit_writer_bytes_written(&bw);
    size_t cap=(size_t)child->frame_width*child->frame_height+4096u,size=0;
    for (;;) {
        uint8_t* buf=(uint8_t*)realloc(child->packet,cap);
        if (!buf) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        child->packet=buf;child->packet_capacity=cap;
        const int st=svt_av2_tile_writer_write_closed_loop_key_coeffs(h,child->coded_blocks,
            child->coded_block_count,buf,cap,&size,NULL);
        if (st!=SVT_AV2_BW_ERROR_NO_SPACE) {
            if (st || size<=skip || memcmp(buf,header,skip)) return SVT_AV2_ERROR_INTERNAL;
            break;
        }
        if (cap>SIZE_MAX/2u) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        cap*=2u;
    }
    column->payload.data=child->packet+skip;
    column->payload.size=size-skip;
    return SVT_AV2_ERROR_NONE;
}

#if M6_PC_WIENER
/* Restoration flags and entropy contexts restart in each column. Filtering
 * reads the deblocked full frame, including the neighboring tile's halo. */
static SvtAv2Error restore_still_columns(SvtAv2EncoderContext* c, StillColumn* cols, int n,
    const SvtAv2TileCoeffBlock* blocks, size_t count, const uint16_t* guide, SvtAv2FrameHeaderConfig* header) {
    const uint32_t w=c->frame_width, h=c->frame_height, depth=c->config.bit_depth, step=depth>8u?2u:1u;
    uint8_t* skip=(uint8_t*)calloc((size_t)w*h/16u,1);
    uint16_t* filtered=(uint16_t*)malloc((size_t)w*h*sizeof(*filtered));
    if (!skip || !filtered) { free(skip); free(filtered); return SVT_AV2_ERROR_OUT_OF_MEMORY; }
    pc_wiener_skip_map(skip,blocks,count,w/4u,h);
    const int status=svt_av2_pcwiener_filter_y_parallel(guide,w,filtered,w,w,h,depth,header->base_q_idx,skip,w/4u,NULL,SVT_AV2_MAX(1u,c->config.threads));
    free(skip);
    if (status) { free(filtered); return SVT_AV2_ERROR_INVALID_STATE; }
    uint32_t ru=256u;
    for (int i=1;i<n;++i) while (cols[i].x%ru) ru/=2u;
    uint32_t any=0;
    for (int i=0;i<n;++i) {
        SvtAv2EncoderContext* child=(SvtAv2EncoderContext*)cols[i].handle;
        SvtAv2FrameHeaderConfig* local=&child->still_tools_header;
        const uint32_t width=child->frame_width, nx=SVT_AV2_MAX(1u,(width+ru/2u)/ru);
        const uint32_t ny=SVT_AV2_MAX(1u,(h+ru/2u)/ru);
        local->pc_wiener_unit=ru;
        for (uint32_t row=0;row<ny;++row) for (uint32_t col=0;col<nx;++col) {
            const uint32_t left=cols[i].x+col*ru, right=cols[i].x+(col+1u==nx?width:(col+1u)*ru);
            const uint32_t top=row?row*ru-8u:0u, bottom=row+1u==ny?h:(row+1u)*ru-8u;
            uint64_t before=0, after=0;
            for (uint32_t y=top;y<bottom;++y) for (uint32_t x=left;x<right;++x) {
                const int orig=svt_av2_sample_load(c->frame_source.planes[0]+(size_t)y*c->frame_source.strides[0]+x*step,depth);
                const int e0=orig-guide[(size_t)y*w+x], e1=orig-filtered[(size_t)y*w+x];
                before+=(uint64_t)(e0*e0); after+=(uint64_t)(e1*e1);
            }
            int use=after<before;
#if M6_CONTENT_GUARD
            if (use && content_guard_domain(c)) {
                M6ContentFilterGuardStats stats;
                use=m6_content_filter_accept(c->frame_source.planes[0],c->frame_source.strides[0],depth,
                    guide,filtered,w,h,left,top,right-left,bottom-top,&stats);
            }
#endif
            if (use) {
                any=1;
                for (uint32_t y=top;y<bottom;++y) for (uint32_t x=left;x<right;++x)
                    svt_av2_sample_store(c->recon.planes[0]+(size_t)y*c->recon.strides[0]+x*step,depth,
                        filtered[(size_t)y*w+x]);
            } else {
                const uint32_t index=row*nx+col;
                local->pc_wiener_none_mask[index/64u] |= UINT64_C(1)<<(index%64u);
            }
        }
    }
    free(filtered);
    header->enable_pc_wiener=any;
    header->pc_wiener_unit=ru;
    for (int i=0;i<n;++i) ((SvtAv2EncoderContext*)cols[i].handle)->still_tools_header.enable_pc_wiener=any;
    return SVT_AV2_ERROR_NONE;
}
#endif

/* The first and last GDF stripes are clipped at the picture boundary.
 * The decoder shifts intermediate filtering-unit rows up by eight pixels. */
static SvtAv2Error select_column_gdf(SvtAv2EncoderContext* c, StillColumn* cols,
    int n, const uint16_t* before, SvtAv2FrameHeaderConfig* header) {
    const uint32_t w=c->frame_width,h=c->frame_height,bd=c->config.bit_depth,step=bd>8u?2u:1u;
    const uint32_t rows=(h+63u)/64u;
    for (int i=0;i<n;++i) {
        SvtAv2EncoderContext* child=(SvtAv2EncoderContext*)cols[i].handle;
        const uint32_t nx=(child->frame_width+63u)/64u;
        cols[i].gdf_flags=(uint8_t*)malloc((size_t)nx*rows);
        if (!cols[i].gdf_flags) return SVT_AV2_ERROR_OUT_OF_MEMORY;
        for (uint32_t sy=0;sy<rows;++sy) for (uint32_t sx=0;sx<nx;++sx) {
            const uint32_t left=cols[i].x+64u*sx;
            const uint32_t right=cols[i].x+SVT_AV2_MIN(child->frame_width,64u*(sx+1u));
            const uint32_t top=sy ? 64u*sy-8u : 0u;
            const uint32_t bottom=sy+1u==rows ? h : 64u*(sy+1u)-8u;
            int64_t gain=0;
            for (uint32_t y=top;y<bottom;++y) for (uint32_t x=left;x<right;++x) {
                const int source=svt_av2_sample_load(c->frame_source.planes[0]+(size_t)y*c->frame_source.strides[0]+x*step,bd);
                const int unfiltered=source-before[(size_t)y*w+x];
                const int filtered=source-svt_av2_sample_load(c->recon.planes[0]+(size_t)y*c->recon.strides[0]+x*step,bd);
                gain+=unfiltered*unfiltered-filtered*filtered;
            }
            cols[i].gdf_flags[sy*nx+sx]=(uint8_t)(gain>0);
            if (gain<=0) for (uint32_t y=top;y<bottom;++y) for (uint32_t x=left;x<right;++x)
                svt_av2_sample_store(c->recon.planes[0]+(size_t)y*c->recon.strides[0]+x*step,bd,before[(size_t)y*w+x]);
        }
        child->still_tools_header.gdf_flags=cols[i].gdf_flags;
        child->still_tools_header.enable_gdf=1;
        child->still_tools_header.gdf_code=header->gdf_code;
    }
    header->gdf_flags=cols[0].gdf_flags;
    return SVT_AV2_ERROR_NONE;
}

static SvtAv2Error encode_still_columns(SvtAv2EncoderContext* c,uint32_t qindex,size_t* bytes) {
    const int n=(int)SVT_AV2_MAX(1u,c->tile_columns);
    const uint32_t sb=c->frame_width/64u, base=sb/(uint32_t)n, extra=sb%(uint32_t)n;
    StillColumn cols[8]={{0}};
    SvtAv2ColumnPayload payloads[8];
    for (int i=0;i<n;++i) cols[i].x=64u*((uint32_t)i*base+SVT_AV2_MIN((uint32_t)i,extra));
    int i;
#ifdef _OPENMP
    const int threads=(int)SVT_AV2_MIN((uint32_t)n,SVT_AV2_MAX(1u,c->config.threads));
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (i=0;i<n;++i) {
        const uint32_t end=i+1<n?cols[i+1].x:c->frame_width;
        cols[i].error=encode_still_column(c,&cols[i],end-cols[i].x,qindex);
    }

    SvtAv2Error err=SVT_AV2_ERROR_NONE;
    size_t capacity=4096u,count=0;
    for (i=0;i<n;++i) {
        if (cols[i].error) { err=cols[i].error; goto done; }
        const SvtAv2EncoderContext* child=(const SvtAv2EncoderContext*)cols[i].handle;
        count+=child->coded_block_count;
        c->rd_stats.candidate_count+=child->rd_stats.candidate_count;
        c->rd_stats.forward_calls+=child->rd_stats.forward_calls;
        c->rd_stats.replay_checks+=child->rd_stats.replay_checks;
        c->rd_stats.distortion+=child->rd_stats.distortion;
        c->rd_stats.rate_q12+=child->rd_stats.rate_q12;
        c->rd_stats.cost+=child->rd_stats.cost;
        c->rd_stats.full_superblocks+=child->rd_stats.full_superblocks;
        c->rd_stats.partial_superblocks+=child->rd_stats.partial_superblocks;
        c->rd_stats.motion_candidates+=child->rd_stats.motion_candidates;
        c->rd_stats.coeff_trials+=child->rd_stats.coeff_trials;
        c->rd_stats.coeff_accepts+=child->rd_stats.coeff_accepts;
        c->rd_stats.motion_trials+=child->rd_stats.motion_trials;
        c->rd_stats.motion_accepts+=child->rd_stats.motion_accepts;
        c->rd_stats.wide_trials+=child->rd_stats.wide_trials;
        c->rd_stats.wide_accepts+=child->rd_stats.wide_accepts;
        c->rd_stats.lambda_q8=child->rd_stats.lambda_q8;
        c->rd_stats.winner_hash=(c->rd_stats.winner_hash^child->rd_stats.winner_hash)*UINT64_C(1099511628211);
    }
    SvtAv2FrameHeaderConfig header=((SvtAv2EncoderContext*)cols[0].handle)->still_tools_header;
    header.frame_width=c->frame_width;header.frame_height=c->frame_height;
    header.frame_width_bits_minus_1=header.frame_height_bits_minus_1=0;
    header.tile_cols_log2=n==1?0u:n==2?1u:n==4?2u:3u;
    header.partition_decisions=NULL; header.partition_decision_count=0;
    header.gdf_code=gdf_key_code(c); header.enable_gdf=header.gdf_code!=0;
    header.deblock_y=3u;header.deblock_delta=header.deblock_delta_h=1;
    /* Filtering crosses tile boundaries. Assemble the decoded block map before
     * running the same full-picture deblock and GDF implementations. */
    SvtAv2TileCoeffBlock* blocks=(SvtAv2TileCoeffBlock*)malloc(count*sizeof(*blocks));
    if (!blocks) { err=SVT_AV2_ERROR_OUT_OF_MEMORY; goto done; }
    const uint32_t sb_cols=(c->frame_width+63u)/64u,sb_rows=(c->frame_height+63u)/64u;
    uint16_t* sb_qindex=NULL;
    if (c->still_aq) {
        sb_qindex=(uint16_t*)malloc((size_t)sb_cols*sb_rows*sizeof(*sb_qindex));
        if (!sb_qindex) { free(blocks); err=SVT_AV2_ERROR_OUT_OF_MEMORY; goto done; }
    }
    size_t offset=0;
    for (i=0;i<n;++i) {
        const SvtAv2EncoderContext* child=(const SvtAv2EncoderContext*)cols[i].handle;
        if (sb_qindex) {
            const uint32_t tile_cols=(child->frame_width+63u)/64u;
            for (uint32_t row=0;row<sb_rows;++row)
                memcpy(sb_qindex+row*sb_cols+cols[i].x/64u,cols[i].qindex+row*tile_cols,
                    tile_cols*sizeof(*sb_qindex));
        }
        for (size_t k=0;k<child->coded_block_count;++k) {
            blocks[offset]=child->coded_blocks[k];
            blocks[offset].x+=cols[i].x>>(blocks[offset].plane!=0);
            ++offset;
        }
    }
    uint64_t edges,changed,tile_edges=0u;
    for (i=1;i<n;++i) tile_edges|=UINT64_C(1)<<(cols[i].x/64u);
    err=svt_av2_deblock_luma_columns(&c->recon,blocks,count,3u,1,1,qindex,sb_qindex,tile_edges,&edges,&changed);
    free(sb_qindex);
    uint16_t* guide=NULL;
    const char* restoration=getenv("SVT_AV2_TILE_RESTORATION");
    const int restore=restoration && !strcmp(restoration,"pcw");
    if (!err && (header.gdf_code || restore)) {
        guide=svt_av2_gdf_copy_luma(&c->recon);
        if (!guide) err=SVT_AV2_ERROR_OUT_OF_MEMORY;
    }
#if M6_PC_WIENER
    if (!err && restore) err=restore_still_columns(c,cols,n,blocks,count,guide,&header);
#else
    if (restore) err=SVT_AV2_ERROR_UNSUPPORTED;
#endif
    free(blocks);
    uint16_t* restored=NULL;
    if (!err && restore && c->block_gdf && header.gdf_code) {
        restored=svt_av2_gdf_copy_luma(&c->recon);
        if (!restored) err=SVT_AV2_ERROR_OUT_OF_MEMORY;
    }
    const char* gdf_search=getenv("SVT_AV2_GDF_SEARCH");
    if (!err && header.gdf_code && gdf_search && !strcmp(gdf_search,"sampled")) {
        err=svt_av2_gdf_select(&c->recon,&c->frame_source,guide,qindex,0u,&header.gdf_code);
        header.enable_gdf=header.gdf_code!=0;
    }
    if (!err && header.gdf_code)
        err=svt_av2_gdf_luma(&c->recon,guide,guide,qindex,header.gdf_code,0u,SVT_AV2_MAX(1u,c->config.threads));
    if (!err && c->block_gdf && header.gdf_code)
        err=select_column_gdf(c,cols,n,restored ? restored : guide,&header);
    free(restored);
    free(guide);
    if (err) goto done;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (i=0;i<n;++i) cols[i].error=pack_still_column(&cols[i]);
    for (i=0;i<n;++i) {
        if (cols[i].error) { err=cols[i].error; goto done; }
        payloads[i]=cols[i].payload;
        capacity+=payloads[i].size+4u;
    }
    if (c->packet_capacity<capacity) {
        uint8_t* buf=(uint8_t*)realloc(c->packet,capacity);
        if (!buf) { err=SVT_AV2_ERROR_OUT_OF_MEMORY; goto done; }
        c->packet=buf;c->packet_capacity=capacity;
    }
    if (svt_av2_pack_still_columns(&header,payloads,c->packet,c->packet_capacity,bytes)) err=SVT_AV2_ERROR_INTERNAL;
done:
    for (i=0;i<n;++i) {if (cols[i].handle) svt_av2_enc_deinit(&cols[i].handle);free(cols[i].qindex);free(cols[i].gdf_flags);}
    return err;
}
