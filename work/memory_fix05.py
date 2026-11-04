"""Execution-only repair: bounded, resumable monthly aggregation for frozen Fix05."""
from datetime import date
from pathlib import Path
import gc,hashlib,json,os,shutil,uuid


def file_hash05m(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def sql_literal05m(value):
    return "'"+str(value).replace("'","''")+"'"


def parquet05m(paths):
    return 'read_parquet(['+','.join(sql_literal05m(Path(p).as_posix() if isinstance(p,Path) else p) for p in paths)+'],hive_partitioning=true)'


def month_bounds05m(month):
    year,number=map(int,month.split('-'))
    start=date(year,number,1)
    end=date(year+1,1,1) if number==12 else date(year,number+1,1)
    return start,end


def memory_settings05m(con,root):
    """Leave space for Python and allocations outside DuckDB's memory manager."""
    gc.collect(); candidates=[]
    try:
        info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        candidates.append(int(info['MemAvailable'].strip().split()[0])*1024)
    except (OSError,KeyError,ValueError):pass
    for limit_path,used_path in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                 ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            limit=Path(limit_path).read_text().strip()
            if limit!='max':candidates.append(max(0,int(limit)-int(Path(used_path).read_text().strip())))
        except (OSError,ValueError):pass
    available=min(candidates) if candidates else None
    if available is not None and available<1024**3:
        raise RuntimeError('Less than 1 GiB available RAM. Save the notebook, restart the session, then use the memory-safe standalone notebook.')
    mib=min(4096,max(512,int(.4*available/1024**2))) if available else 1536
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<2*1024**3:
        raise RuntimeError('Less than 2 GiB free disk for checkpoints/spill. Free space outside the verified recovery cache before retrying.')
    con.execute(f"SET memory_limit='{mib}MiB'")
    con.execute('SET threads=1')
    con.execute('SET preserve_insertion_order=false')
    con.execute('SET temp_directory='+sql_literal05m((root/'spill').resolve().as_posix()))
    return {'duckdb_memory_limit_mib':mib,'threads':1,'preserve_insertion_order':False,
            'available_memory_bytes_at_configuration':available}


def month_sql05m(source,month=None):
    # Every offset is relative to its own 28-day window. Exact bit sets replace
    # two large COUNT(DISTINCT ...) states per page/window; absence stays absent.
    scan=parquet05m(source)
    checked_date='d.report_date'
    if month is not None:
        start,end=month_bounds05m(month)
        # Validate in the expression used by the join, so out-of-window corrupt
        # dates cannot disappear before the audit. No separate remote date scan.
        scan=f"""(SELECT *,CASE WHEN report_date IS NOT NULL AND
          report_date>=DATE '{start}' AND report_date<DATE '{end}' THEN report_date
          ELSE error('Source partition contains null/out-of-month dates; stop before combining months.')
          END AS _checked_report_date FROM {scan})"""
        checked_date='d._checked_report_date'
    return f"""WITH grouped AS (
      SELECT w.origin,w.role,d.client_hash_id,d.content_hash_id,COUNT(*) AS raw_rows,
        bit_or(1::BIGINT << CAST(date_diff('day',w.start_date,{checked_date}) AS INTEGER)) AS row_mask,
        bit_or(CASE WHEN gsc_data_available IS TRUE THEN
          1::BIGINT << CAST(date_diff('day',w.start_date,{checked_date}) AS INTEGER)
          ELSE 0::BIGINT END) AS available_mask,
        COUNT(*) FILTER(WHERE gsc_data_available IS TRUE AND
          (gsc_impressions IS NULL OR gsc_clicks IS NULL OR
           NOT isfinite(gsc_impressions) OR NOT isfinite(gsc_clicks) OR
           gsc_impressions<0 OR gsc_clicks<0 OR gsc_clicks>gsc_impressions OR
           gsc_impressions!=floor(gsc_impressions) OR gsc_clicks!=floor(gsc_clicks))) AS bad_counts,
        COUNT(*) FILTER(WHERE gsc_data_available IS TRUE AND gsc_impressions>0 AND
          (gsc_sum_position IS NULL OR NOT isfinite(gsc_sum_position) OR gsc_sum_position<0)) AS bad_position,
        CAST(SUM(gsc_impressions) FILTER(WHERE gsc_data_available IS TRUE) AS BIGINT) AS impressions,
        CAST(SUM(gsc_clicks) FILTER(WHERE gsc_data_available IS TRUE) AS BIGINT) AS clicks,
        SUM(gsc_sum_position) FILTER(WHERE gsc_data_available IS TRUE) AS sum_position
      FROM {scan} d
      JOIN _fix05_windows w ON {checked_date} BETWEEN w.start_date AND w.end_date
      GROUP BY 1,2,3,4)
      SELECT *,CAST(bit_count(row_mask) AS BIGINT) AS raw_days,
        CAST(bit_count(available_mask) AS BIGINT) AS available_days FROM grouped"""


def merge_sql05m(paths,origin,role):
    return f"""WITH grouped AS (
      SELECT origin,role,client_hash_id,content_hash_id,
        CAST(SUM(raw_rows) AS BIGINT) AS raw_rows,
        bit_or(row_mask) AS row_mask,bit_or(available_mask) AS available_mask,
        CAST(SUM(raw_days) AS BIGINT) AS partial_day_total,
        CAST(SUM(bad_counts) AS BIGINT) AS bad_counts,
        CAST(SUM(bad_position) AS BIGINT) AS bad_position,
        CAST(SUM(impressions) AS BIGINT) AS impressions,
        CAST(SUM(clicks) AS BIGINT) AS clicks,SUM(sum_position) AS sum_position
      FROM {parquet05m(paths)}
      WHERE origin={sql_literal05m(origin)} AND role={sql_literal05m(role)}
      GROUP BY 1,2,3,4)
      SELECT *,CAST(bit_count(row_mask) AS BIGINT) AS raw_days,
        CAST(bit_count(available_mask) AS BIGINT) AS available_days FROM grouped"""


def validate_aggregate05m(con,path,merged=False):
    extra=' OR partial_day_total!=raw_days' if merged else ''
    n,bad=con.sql(f"""SELECT COUNT(*),COUNT(*) FILTER(WHERE
      client_hash_id IS NULL OR content_hash_id IS NULL OR raw_rows!=raw_days OR
      raw_days!=bit_count(row_mask) OR available_days!=bit_count(available_mask) OR
      raw_days>28 OR row_mask>=268435456 OR available_days>raw_days OR
      (row_mask | available_mask)!=row_mask OR bad_counts>0{extra})
      FROM {parquet05m([path])}""").fetchone()
    if bad:raise RuntimeError('Aggregate coverage/grain/count audit failed. Invalid groups: '+str(bad))
    return int(n)


def checked_checkpoint05m(path,identity):
    path=Path(path);receipt=path.with_suffix('.json')
    if not receipt.exists():return None
    metadata=json.loads(receipt.read_text())
    if (not path.exists() or any(metadata.get(k)!=v for k,v in identity.items()) or
        metadata.get('sha256')!=file_hash05m(path)):
        raise RuntimeError('Recovery checkpoint identity/checksum mismatch; preserve files and investigate.')
    return metadata


def write_checkpoint05m(con,query,path,identity,merged=False,validate=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.stem+'.partial-'+uuid.uuid4().hex+'.parquet')
    con.execute('COPY ('+query+') TO '+sql_literal05m(partial.resolve().as_posix())+' (FORMAT PARQUET, COMPRESSION ZSTD)')
    rows=validate_aggregate05m(con,partial,merged) if validate else None
    metadata={**identity,'sha256':file_hash05m(partial),'n_aggregate_rows':rows}
    if path.exists():
        # Unverified interrupted outputs are preserved, never consumed as checkpoints.
        path.replace(path.with_name(path.stem+'.unverified-'+uuid.uuid4().hex+'.parquet'))
    partial.replace(path)
    receipt=path.with_suffix('.json');pending=receipt.with_suffix('.json.partial')
    pending.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');pending.replace(receipt)
    return metadata


def build_cache05m(con,windows,months,root,protocol_sha,implementation_sha,source_paths,current_revision,progress=print):
    """Bound remote aggregation by month; local merge by window; durable checksums."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    identity={'protocol_sha256':protocol_sha,'implementation_sha256':implementation_sha,'source_months':list(months)}
    plan_path=root/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if any(plan.get(k)!=v for k,v in identity.items()):
            raise RuntimeError('Recovery plan differs from verified checkpoints. Do not mix implementations or protocols.')
    else:
        plan={**identity,'dataset_revision':current_revision()}
        plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    identity={**identity,'dataset_revision':plan['dataset_revision']}
    final=root/'windows.parquet'
    final_identity={**identity,'kind':'combined_verified_windows'}
    ready=checked_checkpoint05m(final,final_identity)
    if ready:
        progress('Reusing verified recovery cache; no warehouse scan.')
        return final,ready
    month_paths=[];month_receipts=[]
    for number,month in enumerate(months,1):
        path=root/'months'/f'{month}.parquet';part_identity={**identity,'kind':'month','source_month':month}
        receipt=checked_checkpoint05m(path,part_identity)
        if receipt is None:
            if current_revision()!=plan['dataset_revision']:
                raise RuntimeError('Warehouse revision differs from existing checkpoints; mixed revisions are forbidden.')
            start,end=month_bounds05m(month);sources=source_paths(month)
            selected=windows.loc[(windows.start_date<end)&(windows.end_date>=start)]
            if not len(selected):raise ValueError('Frozen month has no intersecting windows.')
            con.register('_fix05_windows',selected)
            progress(f'Month {number}/{len(months)}: aggregating {month}; completed months are checkpointed.')
            # Stage the file first. A revision change must not bless its receipt.
            stage=root/'pending'/f'{month}.parquet';stage.parent.mkdir(parents=True,exist_ok=True)
            staged=write_checkpoint05m(con,month_sql05m(sources,month),stage,part_identity)
            if current_revision()!=plan['dataset_revision']:
                raise RuntimeError('Warehouse revision changed during this month; pending files will not be reused.')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():path.replace(path.with_name(path.stem+'.unverified-'+uuid.uuid4().hex+'.parquet'))
            stage.replace(path);stage.with_suffix('.json').replace(path.with_suffix('.json'))
            receipt=staged
        else:progress(f'Month {number}/{len(months)}: reusing verified {month} checkpoint.')
        month_paths.append(path);month_receipts.append(receipt)
    window_paths=[]
    for _,window in windows.iterrows():
        origin,role=window.origin,window.role
        path=root/'windows'/f'{origin}-{role}.parquet'
        part_identity={**identity,'kind':'window','origin':str(origin),'role':str(role)}
        if checked_checkpoint05m(path,part_identity) is None:
            selected_paths=[p for p,month in zip(month_paths,months)
                if month_bounds05m(month)[0]<=window.end_date and month_bounds05m(month)[1]>window.start_date]
            progress('Merging private local aggregates: '+str(origin)+' '+str(role))
            write_checkpoint05m(con,merge_sql05m(selected_paths,origin,role),path,part_identity,merged=True)
        window_paths.append(path)
    # No global GROUP BY: concatenate verified window files as a streaming COPY.
    metadata=write_checkpoint05m(con,'SELECT * FROM '+parquet05m(window_paths),final,
        {**final_identity,'month_checkpoints':[{'source_month':r['source_month'],'sha256':r['sha256']}
                                              for r in month_receipts]},validate=False)
    return final,metadata
