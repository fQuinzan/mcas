(window.webpackJsonp=window.webpackJsonp||[]).push([[49],{"013z":function(e,t,a){"use strict";var r=a("q1tI"),n=a.n(r),b=a("NmYn"),l=a.n(b),i=a("Wbzz"),o=a("Xrax"),c=a("k4MR"),s=a("TSYQ"),u=a.n(s),d=a("QH2O"),m=a.n(d),p=a("qKvR"),g=function(e){var t,a=e.title,r=e.theme,n=e.tabs,b=void 0===n?[]:n;return Object(p.b)("div",{className:u()(m.a.pageHeader,(t={},t[m.a.withTabs]=b.length,t[m.a.darkMode]="dark"===r,t))},Object(p.b)("div",{className:"bx--grid"},Object(p.b)("div",{className:"bx--row"},Object(p.b)("div",{className:"bx--col-lg-12"},Object(p.b)("h1",{id:"page-title",className:m.a.text},a)))))},j=a("BAC9"),O=function(e){var t=e.relativePagePath,a=e.repository,r=Object(i.useStaticQuery)("1364590287").site.siteMetadata.repository,n=a||r,b=n.baseUrl,l=n.subDirectory,o=b+"/edit/"+n.branch+l+"/src/pages"+t;return b?Object(p.b)("div",{className:"bx--row "+j.row},Object(p.b)("div",{className:"bx--col"},Object(p.b)("a",{className:j.link,href:o},"Edit this page on GitHub"))):null},h=a("FCXl"),x=a("dI71"),f=a("I8xM"),w=function(e){function t(){return e.apply(this,arguments)||this}return Object(x.a)(t,e),t.prototype.render=function(){var e=this.props,t=e.title,a=e.tabs,r=e.slug,n=r.split("/").filter(Boolean).slice(-1)[0],b=a.map((function(e){var t,a=l()(e,{lower:!0,strict:!0}),b=a===n,o=new RegExp(n+"/?(#.*)?$"),c=r.replace(o,a);return Object(p.b)("li",{key:e,className:u()((t={},t[f.selectedItem]=b,t),f.listItem)},Object(p.b)(i.Link,{className:f.link,to:""+c},e))}));return Object(p.b)("div",{className:f.tabsContainer},Object(p.b)("div",{className:"bx--grid"},Object(p.b)("div",{className:"bx--row"},Object(p.b)("div",{className:"bx--col-lg-12 bx--col-no-gutter"},Object(p.b)("nav",{"aria-label":t},Object(p.b)("ul",{className:f.list},b))))))},t}(n.a.Component),v=a("MjG9"),N=a("CzIb");t.a=function(e){var t=e.pageContext,a=e.children,r=e.location,n=e.Title,b=t.frontmatter,s=void 0===b?{}:b,u=t.relativePagePath,d=t.titleType,m=s.tabs,j=s.title,x=s.theme,f=s.description,P=s.keywords,T=Object(N.a)().interiorTheme,k=Object(i.useStaticQuery)("2456312558").site.pathPrefix,y=k?r.pathname.replace(k,""):r.pathname,C=m?y.split("/").filter(Boolean).slice(-1)[0]||l()(m[0],{lower:!0}):"",S=x||T;return Object(p.b)(c.a,{tabs:m,homepage:!1,theme:S,pageTitle:j,pageDescription:f,pageKeywords:P,titleType:d},Object(p.b)(g,{title:n?Object(p.b)(n,null):j,label:"label",tabs:m,theme:S}),m&&Object(p.b)(w,{title:j,slug:y,tabs:m,currentTab:C}),Object(p.b)(v.a,{padded:!0},a,Object(p.b)(O,{relativePagePath:u})),Object(p.b)(h.a,{pageContext:t,location:r,slug:y,tabs:m,currentTab:C}),Object(p.b)(o.a,null))}},BAC9:function(e,t,a){e.exports={bxTextTruncateEnd:"EditLink-module--bx--text-truncate--end--2pqje",bxTextTruncateFront:"EditLink-module--bx--text-truncate--front--3_lIE",link:"EditLink-module--link--1qzW3",row:"EditLink-module--row--1B9Gk"}},I8xM:function(e,t,a){e.exports={bxTextTruncateEnd:"PageTabs-module--bx--text-truncate--end--267NA",bxTextTruncateFront:"PageTabs-module--bx--text-truncate--front--3xEQF",tabsContainer:"PageTabs-module--tabs-container--8N4k0",list:"PageTabs-module--list--3eFQc",listItem:"PageTabs-module--list-item--nUmtD",link:"PageTabs-module--link--1mDJ1",selectedItem:"PageTabs-module--selected-item--YPVr3"}},QH2O:function(e,t,a){e.exports={bxTextTruncateEnd:"PageHeader-module--bx--text-truncate--end--mZWeX",bxTextTruncateFront:"PageHeader-module--bx--text-truncate--front--3zvrI",pageHeader:"PageHeader-module--page-header--3hIan",darkMode:"PageHeader-module--dark-mode--hBrwL",withTabs:"PageHeader-module--with-tabs--3nKxA",text:"PageHeader-module--text--o9LFq"}},tpJo:function(e,t,a){"use strict";a.r(t),a.d(t,"_frontmatter",(function(){return i})),a.d(t,"default",(function(){return s}));var r=a("wx14"),n=a("zLVn"),b=(a("q1tI"),a("7ljp")),l=a("013z"),i=(a("qKvR"),{}),o={_frontmatter:i},c=l.a;function s(e){var t=e.components,a=Object(n.a)(e,["components"]);return Object(b.b)(c,Object(r.a)({},o,a,{components:t,mdxType:"MDXLayout"}),Object(b.b)("h2",null,"Presentations"),Object(b.b)("ul",null,Object(b.b)("li",{parentName:"ul"},Object(b.b)("a",{parentName:"li",href:"https://drive.google.com/file/d/1HqUiIycuBfaEmjCxHRaHpu8UpVsmePyk/view?usp=sharing"},"Overview of MCAS (September 2020)"))),Object(b.b)("h2",null,"Papers"),Object(b.b)("ul",null,Object(b.b)("li",{parentName:"ul"},Object(b.b)("a",{parentName:"li",href:"https://arxiv.org/abs/2103.00007"},"An Architecture for Memory Centric Active Storage (MCAS)"))),Object(b.b)("h2",null,"Podcasts"),Object(b.b)("h3",null,"Storage Unpacked Podcast - December 2020"),Object(b.b)("ul",null,Object(b.b)("li",{parentName:"ul"},Object(b.b)("a",{parentName:"li",href:"https://storageunpacked.com/2020/11/184-mcas/"},"#184 MCAS - Memory Centric Active Storage"))),Object(b.b)("h2",null,"Videos"),Object(b.b)("h3",null,"SNIA Storage Developers Conference (SDC2020) - September 2020"),Object(b.b)("ul",null,Object(b.b)("li",{parentName:"ul"},"SNIA SDC web site - ",Object(b.b)("a",{parentName:"li",href:"https://www.snia.org/events/storage-developer"},"https://www.snia.org/events/storage-developer")),Object(b.b)("li",{parentName:"ul"},"Exploring New Storage Paradigms and Opportunities with Persistent Memory - ",Object(b.b)("a",{parentName:"li",href:"https://youtu.be/ftj73Nlefao"},"https://youtu.be/ftj73Nlefao")),Object(b.b)("li",{parentName:"ul"},Object(b.b)("a",{parentName:"li",href:"https://storagedeveloper2020.org/sites/default/files/004-Waddington-Exploring-New-Storage-Paradigms.pdf"},"Talk presentation material"))),Object(b.b)("h3",null,"ISPASS 2020 Conference - August 2020"),Object(b.b)("ul",null,Object(b.b)("li",{parentName:"ul"},"Conference presentation of the high-performance key-value store element of MCAS\n",Object(b.b)("a",{parentName:"li",href:"https://www.youtube.com/watch?v=9uc4xoPgG6c&list=PLHJB2bhmgB7crXM7wBKIDi7OEa0UTZtrR&index=17"},"https://www.youtube.com/watch?v=9uc4xoPgG6c&list=PLHJB2bhmgB7crXM7wBKIDi7OEa0UTZtrR&index=17"))))}s.isMDXComponent=!0}}]);
//# sourceMappingURL=component---src-pages-news-index-mdx-dd65a33e4373d426c887.js.map