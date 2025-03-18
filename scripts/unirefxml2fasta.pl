#process uniref xml or its gz file and extract sequences in fasta
#does not handle any other format of xmls 
#Since it will process huge xml files, parsing is done linearly as text format. Order of entries will be exactly the same as in the xml. (i.e. different from released fasta files, which are generated from internal data)

#perl unirefxml2fasta.pl XML_FILE FASTA_FILE

if(!defined $ARGV[0]){
	print "Run as \"perl unirefxml2fasta.pl uniref100.xml.gz \[uniref100.fasta.gz\]\"\n";
	print "The output file name is automatically generatate if not present\n";
	exit;
}
#handle input output files
$XML_FILE=$ARGV[0];
$FASTA_FILE="extract.fasta";
$LOG_FILE="extract.log";
if($XML_FILE !~ /\.gz$/){
	open XML,$XML_FILE or die "cannot open file $XML_FILE";
	$XML_FILE =~ s/\.xml$//;
}else{
	open XML,"gunzip -c $XML_FILE |" or die "cannot open gzip file $XML_FILE";
	$XML_FILE =~ s/\.gz$//;	$XML_FILE =~ s/\.xml$//;
}
if(!defined $ARGV[1]){ #no default name
	$FASTA_FILE="$XML_FILE.fasta.gz";
	$LOG_FILE="$XML_FILE.logs";
}else{
	$FASTA_FILE=$ARGV[1];
	$LOG_FILE="$FASTA_FILE.logs";
}
#parse content: entry id, cluster name, member count, common taxon, and representative id
#<entry id="UniRef100_Q6GZX4" updated="2015-07-22">
#<name>Cluster: Putative transcription factor 001R</name>
#<property type="member count" value="2"/>
#<property type="common taxon" value="Frog virus 3"/>
#...
#<representativeMember>
#  <dbReference type="UniProtKB ID" id="001R_FRG3G">
#...
#  <sequence length="256" checksum="B4840739BF7D4121">
#MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS
#...
#>UniRef100_Q6GZX4 Putative transcription factor 001R n=2 Tax=Frog virus 3 RepID=001R_FRG3G
open OUT,"| gzip -c >$FASTA_FILE";
#open LOG,">$LOG_FILE";
print "writing to file $FASTA_FILE\n";

#parse the xml
$iLinenum=0;	$iEntrynum=0;	$iMaxErrcnt=1000; $iErrcnt=0;
$iEntrynum=$iErrEntrycnt=$iErrLengthcnt=$iErrKBcnt=$iErrPKcnt=$iErrMemcnt=$max_mem=0;
$clstr_name=$mem_cnt=$common_taxon=$common_taxon_id=$repid="N\/A";
$stage=9999;

#decide xml file version for auto convert
#releaseDate="2020-04-22" version="2020_02">
my $xml_version="0000_00"; my $rel_year=0; my $rel_vers=0; my $reltype=-1;
while($line=<XML>){		chomp($line);	$iLinenum++;
	if($line =~ /releaseDate\s*=\s*\"([\d,\-]+)\"\s*version=\"(\d+)_(\d+)\"\s*>\s*$/){
		$reldate=$1; $rel_year=$2; $rel_vers=$3; $xml_version=$rel_year."_".$rel_vers;
		print "Found xml release version $xml_version, $reldate\n";
		if($rel_year > 2019 || ( $rel_year = 2019 && $rel_vers >= 9) ){
			$reltype=1; #new one liner
			print "\tusing new XML format with one-liner sequence\n";
		}else{
			$reltype=0; #old fasta seq
			print "\tusing old XML format with fasta format, 60 AA\n";
		}
	}elsif($line=~/^<entry\s+id=\"(UniRef\d+_[A-Z0-9]+\-{0,1}\d{0,3})\"\s+updated=\"([\d,\-]+)/){		
		$entry_id=$1;	$date=$2;	
		$iEntrynum++;	$stage=0; $count_len=0; $seq="";
		$clstr_name=$mem_cnt=$common_taxon=$common_taxon_id=$repid="N\/A";
		last;
	}
	print "Header$iLinenum: $line\n";
	if($iLinenum>=5){
		last;
	}
}
if($reltype<0){
	print "The script did not find release version in XML header (like version=2020_02).\n";
	print "Most likely the XML file did not include the proper header (part of file?)\n";
	print "\nThis is important because of a format change since 2019_09\n";
	print "For any UniREf XML before 2019_09 (not included) please use unirefxml2fasta_old.pl\n";
	print "For any later UniREf XML release version please use unirefxml2fasta_new.pl\n";
	print "Refer to the README file for more specific directions.\n";
	exit;
}

#content start
while($line=<XML>){		chomp($line);	$iLinenum++;
	if($line=~/^<entry\s+id=\"(UniRef\d+_[A-Z0-9]+\-{0,1}\d{0,3})\"\s+updated=\"([\d,\-]+)/){
		$tmp_entry_id=$1;
		&record_entry; 
		$entry_id=$tmp_entry_id;	$date=$2;	
		$iEntrynum++;	$stage=0; $count_len=0; $seq="";
		$clstr_name=$mem_cnt=$common_taxon=$common_taxon_id=$repid="N\/A";
		
	}elsif($stage==0 && $line=~/^\s*<name>Cluster\:\s+([^<]+)<\/name>/){
		$clstr_name=$1; $stage=100;
	}elsif($stage==100 && $line=~/^\s*<property\stype=\"member\s+count\"\s+value=\"([\d]+)\"\/>/){
		$mem_cnt=$1;	$stage=200;
	}elsif($stage==200 && $line=~/^\s*<property\stype=\"common\s+taxon\"\s+value=\"([^\"]+)\"\/>/){
		$common_taxon=$1;	$stage=300;
	}elsif($stage==300 && $line=~/^\s*<property\stype=\"common\s+taxon\s+ID\"\s+value=\"([^\"]+)\"\/>/){
		$common_taxon_id=$1;	$stage=300;
	}elsif($stage==300 && $line=~/^\s*<representativeMember>/){
		$stage=400;
	}elsif($stage==400 && $line=~/^\s*<dbReference\s+type=\"([^\"]+)\"\s+id=\"([\w,\d,\-]+)\">/){
		$type=$1;	$repid=$2;	$stage=500;
	}elsif($stage==500 && $line=~/^\s*<\/dbReference>\s*$/){
		$stage=600;
					
	}elsif($reltype==1 && $stage==600 && $line=~/^\s*<sequence\s+length=\"(\d+)\"\s+checksum=\"([^\"]+)\"\s*>([A-Z]+)</){
		$seqlen=$1;	$checksum=$2; $seq=$3; $stage=800; $count_len=length($seq);
	}elsif($reltype==0 && $stage==600 && $line=~/^\s*<sequence\s+length=\"(\d+)\"\s+checksum/){
		$seqlen=$1;	 $stage=700;
	}elsif($reltype==0 && $stage==700 && $line=~/^\s*([A-Z]+)\s*$/){
		$count_len+=length($line);
		$seq.="$line\n";
	}elsif($reltype==0 && $stage==700 && $line=~/^\s*<\/sequence>\s*$/){
		$stage=800;
	}elsif($stage==800 && $line=~/^\s*<\/representativeMember>\s*$/){
		$stage=900;
	}elsif($stage==900 && $line=~/^\s*<\/entry>\s*$/){
		$stage=1000;
	}
	#last if ($iLinenum == $ARGV[1]);
	#last if ($iErrcnt >= $iMaxErrcnt);
}
&record_entry; 
close OUT;
#statistics
$info="FILE $FASTA_FILE\n";
$info.="$iEntrynum\ttotal entries parsed\n";
print $info;
	
sub record_entry{
	#convert seq from one liner to fasta format
	if($reltype==1){
		($seq_convert,$seqlen_conv)=&conv_seq($seq);
	}else{
		$seq_convert=$seq; $seqlen_conv=$count_len;
	}
	my $badword="N\/A";
	#validity checks
	$bIsErrOccur=0;
	if($stage<1000){ $iErrEntrycnt++;#last entry did not finish normally
			print LOG "\#$iLinenum\t|Error ending entry: $entry_id at stage=$stage\n";
			print LOG "\t>$entry_id $clstr_name n=$mem_cnt Tax=$common_taxon TaxID=$common_taxon_id RepID=$repid\n$seq";
			$bIsErrOccur=1; return;
	}elsif($stage==9999){	return;	}
		
	if($count_len != $seqlen || $seqlen_conv != $seqlen){ $iErrLengthcnt++;	$bIsErrOccur=1;#sequence line missing
	}
	if($type eq "UniProtKB ID" && $repid !~ /^[A-Z0-9]+_{0,1}[A-Z0-9]+\-{0,1}\d{0,3}$/){ $iErrKBcnt++;	$bIsErrOccur=1;
		print LOG "\#$iLinenum\t|Error id format: Type=$type and ID=$repid\n";
	}elsif($type eq "UniParc ID" && $repid !~ /^UPI00[A-Z0-9]+$/){ $iErrPKcnt++;	$bIsErrOccur=1;
		print LOG "\#$iLinenum\t|Error id format: Type=$type and ID=$repid\n";
	}
	if($mem_cnt<=0){ $iErrMemcnt++;	$bIsErrOccur=1;
		print LOG "\#$iLinenum\t|Error member count $mem_cnt\n";
	}	
	if( $common_taxon eq $badword || $repid eq $badword ){ #|| $common_taxon_id eq $badword || $clstr_name eq $badword 
		$iErrTEXTcnt++;	$bIsErrOccur=1;
		print LOG "\#$iLinenum\t|Error available text $common_taxon $common_taxon_id $repid $clstr_name\n";
	}
	$iErrcnt++ if($bIsErrOccur==1);
	$max_mem=$mem_cnt if $max_mem<$mem_cnt;
	
	#output
	$info=">$entry_id $clstr_name n=$mem_cnt Tax=$common_taxon TaxID=$common_taxon_id RepID=$repid\n$seq_convert";
	$info=~s/\&apos;{0,1}/'/g;
	$info=~s/\&amp;{0,1}/&/g;
	$info=~s/\&gt;{0,1}/>/g;
	$info=~s/\&lt;{0,1}/</g;
	$info=~s/\&quot;{0,1}/</g;
	$info=~s/\\\\/\\/g;
	print OUT $info;
	
}		
sub conv_seq{
	my ($seqstr)=@_; my $comp_len=0;
	my $outstr=""; my $seqlen=length($seqstr);
	while($seqlen > 60){
		my $seqline=substr($seqstr, 0, 60);		
		$outstr.="$seqline\n";
		$seqstr=substr($seqstr, 60);		
		$seqlen=length($seqstr);
		$comp_len+=60;		
	}
	$comp_len+=$seqlen;
	$outstr.="$seqstr\n";
	
	return ($outstr,$comp_len);
}
